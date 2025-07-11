import json

import aws_cdk as cdk
from aws_cdk import (
    aws_opensearchserverless as opensearchserverless,
    aws_ec2 as ec2,
    aws_iam as iam,
    CfnOutput,
    Aws,
    Stack,
)
from constructs import Construct


class OpenSearchServerless(Construct):
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        lambda_role_arn_list,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id)

        # Create OpenSearch Serverless Collection
        collection_name = f"{Stack.of(self).stack_name}-collection"
        collection = opensearchserverless.CfnCollection(
            self,
            "OpenSearchCollection",
            name=collection_name,
            type="VECTORSEARCH",
            description="OpenSearch Serverless Collection for content with vector search",
        )

        # Create network policy to allow access from the VPC
        network_policy = opensearchserverless.CfnSecurityPolicy(
            self,
            "NetworkPolicy",
            name=f"network{self.node.addr[:8]}",
            type="network",
            policy=json.dumps(
                [
                    {
                        "Rules": [
                            {
                                "ResourceType": "collection",
                                "Resource": [f"collection/{collection.name}"],
                            },
                        ],
                        # "SourceVPCEs": [vpc_endpoint.attr_id],
                        "AllowFromPublic": True,
                    },
                    {
                        "Rules": [
                            {
                                "ResourceType": "dashboard",
                                "Resource": [f"collection/{collection.name}"],
                            }
                        ],
                        "AllowFromPublic": True,
                    },
                ]
            ),
        )

        # Create encryption policy
        encryption_policy = opensearchserverless.CfnSecurityPolicy(
            self,
            "encryptionPolicy",
            name=f"encryption{self.node.addr[:8]}",
            type="encryption",
            policy=json.dumps(
                {
                    "Rules": [
                        {
                            "ResourceType": "collection",
                            "Resource": [f"collection/{collection.name}"],
                        }
                    ],
                    "AWSOwnedKey": True,
                }
            ),
        )

        # Create data access policy
        allowed_principles = lambda_role_arn_list + [
            f"arn:aws:iam::{Aws.ACCOUNT_ID}:role/Admin",
            "arn:aws:iam::654654199195:user/wayne.chiu@advantech.com"
        ]
        data_access_policy = opensearchserverless.CfnAccessPolicy(
            self,
            "DataAccessPolicy",
            name=f"dataaccess{self.node.addr[:8]}",
            type="data",
            policy=json.dumps(
                [
                    {
                        "Rules": [
                            {
                                "Resource": [f"collection/{collection.name}"],
                                "Permission": ["aoss:*"],
                                "ResourceType": "collection",
                            },
                            {
                                "Resource": [f"index/{collection.name}/*"],
                                "Permission": ["aoss:*"],
                                "ResourceType": "index",
                            },
                        ],
                        "Principal": allowed_principles,
                        "Description": "Allow all access to collection and indexes",
                    }
                ]
            ),
        )

        # Create index using CfnIndex

        # content_index = opensearchserverless.CfnIndex(
        #     self,
        #     "ContentIndex",
        #     collection_endpoint=collection.attr_collection_endpoint,
        #     index_name="content",
        #     # Direct mappings without nested "properties"
        #     mappings={
        #         "content_embedding": {
        #             "type": "knn_vector",
        #             "dimension": 1024,
        #             "method": {
        #                 "name": "hnsw",
        #                 "space_type": "cosinesimil",
        #                 "engine": "nmslib",
        #                 "parameters": {"ef_construction": 512, "m": 16},
        #             },
        #         },
        #         "images.image_embedding": {
        #             "type": "knn_vector",
        #             "dimension": 1024,
        #             "method": {
        #                 "name": "hnsw",
        #                 "space_type": "cosinesimil",
        #                 "engine": "nmslib",
        #                 "parameters": {"ef_construction": 512, "m": 16},
        #             },
        #         },
        #     },
        #     settings={"knn": True, "refresh_interval": "1s"},
        # )

        # Set dependencies
        collection.node.add_dependency(encryption_policy)
        network_policy.node.add_dependency(collection)
        # content_index.node.add_dependency(collection)
        # content_index.node.add_dependency(data_access_policy)

        self.endpoint = collection.attr_collection_endpoint
        self.collection_name = collection.name
        self.collection = collection

        # Outputs
        CfnOutput(self, "CollectionId", value=collection.attr_id)
        CfnOutput(self, "CollectionName", value=collection.name)
        CfnOutput(self, "CollectionEndpoint", value=collection.attr_collection_endpoint)
        CfnOutput(self, "DashboardEndpoint", value=collection.attr_dashboard_endpoint)
        # CfnOutput(self, "IndexUuid", value=content_index.attr_uuid)

    def get_endpoint(self):

        return self.endpoint
