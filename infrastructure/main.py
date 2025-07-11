import os
from aws_cdk import Stack, CfnOutput, Size
from constructs import Construct
from aws_cdk import Stack, RemovalPolicy, Duration, Aws
from aws_cdk import aws_s3 as s3
from aws_cdk import aws_s3_deployment as s3deploy
from aws_cdk import aws_iam as iam
from aws_cdk import aws_lambda as lambda_

from .helpers import provision_lambda_function_with_vpc
from .opensearch_serverless import OpenSearchServerless
from .vpc_stack import VPCStack

DEFAULT_INDEX_NAME = "demo_index"
DEFAULT_VECTOR_FIELD = "image_vector"
DEFAULT_TEXT_FIELD = "image_file"
DEFAULT_METADATA_FIELD = "metadata"


class MyStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        self.data_bucket = self.create_data_bucket("product")

        embedding_model_id = "cohere.embed-multilingual-v3"
        embedding_model_dimension = 1024
        opensearch_index_name = "eccn"

        deployment = s3deploy.BucketDeployment(
            self,
            "DeployFiles",
            sources=[
                s3deploy.Source.asset(
                    os.path.join(os.path.dirname(__file__), "../sample_data")
                )
            ],
            destination_bucket=self.data_bucket,
        )
        # crate a VPC
        vpc_stack = VPCStack(self)
        self.vpc = vpc_stack.get_vpc()

        self.third_parth_layer = self._provision_third_party_layer()

        # create a lambda function in the vpc
        self.ccl_ingestion_lambda = self._provision_lambda_function(
            "ccl_ingestion", vpc_stack.get_vpc(), self.third_parth_layer
        )

        self.find_eccn_lambda = self._provision_lambda_function(
            "find_eccn", vpc_stack.get_vpc(), self.third_parth_layer
        )
        self.data_bucket.grant_read(self.find_eccn_lambda)

        # set up mcp server and mcp client
        # self.mcp_server_lambda = self._provision_mcp_lambda(vpc_stack)

        # self.eccn_lambda = self._provision_eccn_lambda(vpc_stack)

        # set up mcp server all
        # function_url = self.mcp_server_lambda.add_function_url(
        #     auth_type=lambda_.FunctionUrlAuthType.AWS_IAM,  # Public access
        #     cors=lambda_.FunctionUrlCorsOptions(
        #         allowed_origins=["*"],
        #         allowed_methods=[lambda_.HttpMethod.ALL],
        #         allowed_headers=["*"],
        #     ),
        # )
        #
        # self.mcp_server_lambda.grant_invoke_url(self.eccn_lambda.role)
        # self.eccn_lambda.add_environment("MCP_SERVER_URL", function_url.url)

        # Output the function URL
        # CfnOutput(
        #     self,
        #     "ECCNFunctionUrl",
        #     value=function_url.url,
        #     description="URL for the ECCN lookup Lambda function",
        # )

        # create opensearch serverless

        self.oss_collection = OpenSearchServerless(
            self,
            "OSS",
            lambda_role_arn_list=[
                self.ccl_ingestion_lambda.role.role_arn,
                self.find_eccn_lambda.role.role_arn,
            ],
        )

        self.ccl_ingestion_lambda.add_environment(
            "OPENSEARCH_ENDPOINT", self.oss_collection.get_endpoint()
        )
        self.ccl_ingestion_lambda.add_environment(
            "EMBEDDING_MODEL_ID", embedding_model_id
        )

        self.ccl_ingestion_lambda.add_environment(
            "EMBEDDING_MODEL_DIMENSION", str(embedding_model_dimension)
        )

        self.ccl_ingestion_lambda.add_environment(
            "OPENSEARCH_INDEX", opensearch_index_name
        )

        self.find_eccn_lambda.add_environment(
            "OPENSEARCH_ENDPOINT", self.oss_collection.get_endpoint()
        )
        self.find_eccn_lambda.add_environment("EMBEDDING_MODEL_ID", embedding_model_id)

        self.find_eccn_lambda.add_environment("OPENSEARCH_INDEX", opensearch_index_name)

        # self.mcp_server_lambda.add_environment(
        #     "OPENSEARCH_ENDPOINT", self.oss_collection.get_endpoint()
        # )
        #
        # self.mcp_server_lambda.add_environment(
        #     "OPENSEARCH_INDEX", opensearch_index_name
        # )

        # 7. Grant Lambda permissions to access OpenSearch Serverless
        self.ccl_ingestion_lambda.add_to_role_policy(
            iam.PolicyStatement(
                actions=[
                    "aoss:APIAccessAll",  # For API access
                    "aoss:DescribeCollection",
                    "aoss:SearchDocument",
                    "aoss:IndexDocument",
                    "aoss:GetDocument",
                    "aoss:DeleteDocument",
                ],
                resources=[self.oss_collection.collection.attr_arn],
            )
        )

        self.find_eccn_lambda.add_to_role_policy(
            iam.PolicyStatement(
                actions=[
                    "aoss:APIAccessAll",  # For API access
                    "aoss:DescribeCollection",
                    "aoss:SearchDocument",
                    "aoss:IndexDocument",
                    "aoss:GetDocument",
                    "aoss:DeleteDocument",
                ],
                resources=[self.oss_collection.collection.attr_arn],
            )
        )

        # self.mcp_server_lambda.add_to_role_policy(
        #     iam.PolicyStatement(
        #         actions=[
        #             "aoss:APIAccessAll",  # For API access
        #             "aoss:DescribeCollection",
        #             "aoss:SearchDocument",
        #             "aoss:GetDocument",
        #         ],
        #         resources=[self.oss_collection.collection.attr_arn],
        #     )
        # )

    def _provision_third_party_layer(self) -> lambda_.LayerVersion:
        """
        Creates and returns a Lambda layer for third-party dependencies.

        This method provisions a new AWS Lambda Layer, intended to provide
        additional libraries or dependencies external to the Lambda function
        itself. The layer is specifically configured for a Python 3.12 runtime
        and is compatible with x86_64 architecture.

        The layer's content is loaded from a zip file located relative to this
        script's directory, specifically in the 'lambda_layers/third_party' subdirectory.

        Returns:
            lambda_.LayerVersion: An object representing the created Lambda layer,
            ready to be attached to Lambda functions within the same AWS environment.

        Raises:
            FileNotFoundError: If the layer zip file does not exist at the specified path.
            AWS CDK specific exceptions related to resource creation and configuration might also be raised.
        """
        layer = lambda_.LayerVersion(
            self,
            "LLamaIndexLayer",
            compatible_architectures=[lambda_.Architecture.X86_64],
            compatible_runtimes=[lambda_.Runtime.PYTHON_3_10],
            description="llamaindex binary layer",
            code=lambda_.AssetCode(
                os.path.join(
                    os.path.abspath(__file__),
                    "../../src/lambda_layers/third_party/layer.zip",
                )
            ),
        )

        return layer

    def _provision_mcp_lambda(self, vpc_stack):
        """
        Provisions a Lambda function as MCP server.

        Parameters:
            vpc_stack: VPC configuration object where the Lambda will be deployed

        Returns:
            lambda_.Function: Configured Lambda function with VPC access and required permissions
        """

        fn = lambda_.Function(
            self,
            "MCPServer",
            runtime=lambda_.Runtime.NODEJS_22_X,
            handler="dist/index.handler",
            code=lambda_.Code.from_asset(
                os.path.join(os.path.dirname(__file__), "../src/lambdas/mcp_server")
            ),
            vpc=vpc_stack.get_vpc(),
            timeout=Duration.seconds(60),
            memory_size=1024,
        )

        fn.add_to_role_policy(
            iam.PolicyStatement(
                actions=[
                    "bedrock:InvokeModel",
                    "bedrock:InvokeModelWithResponseStream",
                ],
                resources=[
                    f"arn:{Aws.PARTITION}:bedrock:{Aws.REGION}::foundation-model/*"
                ],
                effect=iam.Effect.ALLOW,
            ),
        )

        return fn

    def _provision_eccn_lambda(self, vpc_stack):
        """
        Provisions a Lambda function as MCP server.

        Parameters:
            vpc_stack: VPC configuration object where the Lambda will be deployed

        Returns:
            lambda_.Function: Configured Lambda function with VPC access and required permissions
        """

        fn = lambda_.Function(
            self,
            "ECCNClient",
            runtime=lambda_.Runtime.NODEJS_22_X,
            handler="dist/index.handler",
            code=lambda_.Code.from_asset(
                os.path.join(os.path.dirname(__file__), "../src/lambdas/eccn_client")
            ),
            vpc=vpc_stack.get_vpc(),
            timeout=Duration.seconds(60 * 15),
            memory_size=1024,
        )

        fn.add_to_role_policy(
            iam.PolicyStatement(
                actions=[
                    "bedrock:InvokeModel",
                    "bedrock:InvokeModelWithResponseStream",
                ],
                resources=[
                    f"arn:{Aws.PARTITION}:bedrock:*::foundation-model/*",
                    f"arn:{Aws.PARTITION}:bedrock:*:{Aws.ACCOUNT_ID}:inference-profile/*",
                ],
                effect=iam.Effect.ALLOW,
            ),
        )

        return fn

    def _provision_lambda_function(
        self, function_name, vpc, third_party_layer, description=""
    ):
        """
        Provisions a Lambda function with VPC configuration and third-party layer.

        This method creates and configures a Lambda function with specific VPC settings,
        adds a third-party dependency layer, and sets up necessary IAM permissions for
        Amazon Bedrock access.

        Parameters:
            function_name (str): Name of the Lambda function to be created
            vpc: VPC configuration object where the Lambda will be deployed
            third_party_layer: Lambda layer containing third-party dependencies
            description: lambda function description

        Returns:
            lambda_.Function: Configured Lambda function with VPC access and required permissions

        Example:
            >>> lambda_fn = self._provision_lambda_function("my-function", my_vpc, my_layer)
        """

        fn = provision_lambda_function_with_vpc(
            self,
            function_name,
            vpc,
            description=(
                description
                if description != ""
                else "Lambda function to handle api request"
            ),
            envs={"LOG_LEVEL": "INFO"},
            timeout=900,
            memory_size=1024,
            ephemeral_storage_size=Size.mebibytes(1024),
        )
        fn.add_layers(third_party_layer)

        fn.add_to_role_policy(
            iam.PolicyStatement(
                actions=[
                    "bedrock:InvokeModel",
                    "bedrock:InvokeModelWithResponseStream",
                ],
                resources=[
                    f"arn:{Aws.PARTITION}:bedrock:*::foundation-model/*",
                    f"arn:{Aws.PARTITION}:bedrock:*:{Aws.ACCOUNT_ID}:inference-profile/*",
                ],
                effect=iam.Effect.ALLOW,
            ),
        )

        return fn

    def create_data_bucket(self, bucket_name: str) -> s3.Bucket:
        """
        Creates two S3 buckets: one for storing data and another for storing access logs.

        The data bucket is configured with server-side encryption, SSL enforcement, auto-deletion of objects,
        and a lifecycle rule to expire objects after 180 days. The log bucket is configured to be automatically
        deleted when the stack is destroyed.

        Parameters:
            bucket_name (str): The name of the data bucket to be created. This name must be globally unique.

        Returns:
            s3.Bucket: The newly created data bucket configured with the specified properties.

        Example:
            >>> data_bucket = create_data_bucket("my-data-bucket")
            >>> print(data_bucket.bucket_name)
            "my-data-bucket"
        """
        # create s3 bucket for data storage
        # Bucket for storing logs
        log_bucket = s3.Bucket(
            self, f"LogBucket{bucket_name}", removal_policy=RemovalPolicy.DESTROY
        )

        data_bucket = s3.Bucket(
            self,
            bucket_name,
            encryption=s3.BucketEncryption.S3_MANAGED,
            enforce_ssl=True,
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_objects=True,
            lifecycle_rules=[
                s3.LifecycleRule(
                    enabled=True,
                    expiration=Duration.days(180),
                )
            ],
            server_access_logs_bucket=log_bucket,
            server_access_logs_prefix="access-logs/",
        )

        # Output the bucket name
        CfnOutput(
            self,
            "BucketName",
            value=data_bucket.bucket_name,
            description="Name of the S3 bucket where crawled HTML files can be uploaded for summarization",
        )

        return data_bucket
