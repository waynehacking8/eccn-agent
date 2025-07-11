#   Copyright 2023 Amazon.com and its affiliates; all rights reserved.
#   This file is Amazon Web Services Content and may not be duplicated or distributed without permission.
import hashlib
import os

import aws_cdk

from aws_cdk import (
    aws_iam as iam,
    aws_lambda as lambda_,
    aws_logs as logs,
    custom_resources as cr,
    Stack,
    Duration,
    aws_ec2 as ec2,
)
from constructs import Construct


class CRProvider(Construct):
    def __init__(self, scope: Construct, id: str, props):
        super().__init__(scope, id)

        # Define IAM Role for the Lambda function

        self.role = iam.Role(
            self,
            "CRRole",
            assumed_by=iam.ServicePrincipal("lambda.amazonaws.com"),
            description=f"Role for lambda Resource CR",
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    "service-role/AWSLambdaVPCAccessExecutionRole"
                )
            ],
        )

        for item in props.get("policies", []):
            self.role.add_managed_policy(item)

        # Define the Lambda function for custom resource
        self.custom_resource_function = lambda_.Function(
            self,
            "CustomResourcesFunction",
            code=lambda_.Code.from_asset(props["code_path"]),
            handler=props["handler"],
            runtime=props.get("runtime", lambda_.Runtime.PYTHON_3_10),
            layers=props.get("layers", []),
            role=self.role,
            environment=props.get("envs", {}),
            timeout=Duration.minutes(15),
            vpc=props.get("vpc", None),
            memory_size=128,
            security_groups=props.get("security_groups", None),
            log_retention=logs.RetentionDays.ONE_WEEK,
            description="Custom Resource Provider",
        )

        # Define IAM Role for the Custom Resource Provider
        provider_role = iam.Role(
            self,
            "ProviderRole",
            assumed_by=iam.ServicePrincipal("lambda.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    "service-role/AWSLambdaVPCAccessExecutionRole"
                )
            ],
        )

        # Define the Custom Resource Provider
        self.provider = cr.Provider(
            self,
            "Provider",
            on_event_handler=self.custom_resource_function,
            log_retention=logs.RetentionDays.ONE_WEEK,
            vpc=props.get("vpc", None),
            role=provider_role,
        )

        self.service_token = self.provider.service_token

    @staticmethod
    def get_provider(scope: Construct, provider_name: str, props):
        stack = Stack.of(scope)
        existing = stack.node.try_find_child(provider_name)
        if existing:
            return existing
        return CRProvider(scope, provider_name, props)


def build_custom_resource_provider(props):
    class CRProviderClass:
        def __init__(self, scope: Stack):
            self.provider = CRProvider.get_provider(
                scope, props["provider_name"], props
            )

        @property
        def service_token(self):
            return self.provider.service_token

    return CRProviderClass


def provision_lambda_function_with_vpc(
    construct,
    lambda_fn_name: str,
    vpc: aws_cdk.aws_ec2.Vpc,
    envs={},
    memory_size=128,
    timeout=60,
    description="",
    **kwargs,
) -> lambda_.Function:
    """
    Provisions a Lambda function within the specified VPC's private subnets.

    This function creates a new AWS Lambda function with the provided name and configuration.
    The Lambda function is attached to the private subnets of the given VPC, ensuring that
    it can interact securely with other services within the VPC without exposure to the public internet.

    Args:
        construct (Construct): The CDK construct that serves as the parent of this new Lambda function.
        lambda_fn_name (str): The name to assign to the newly created Lambda function.
        vpc (aws_cdk.aws_ec2.Vpc): The VPC where the Lambda function will be provisioned.
        envs (Optional[Dict[str, str]]): A dictionary containing environment variables to set for the Lambda function.
        memory_size (int): The amount of memory, in MB, allocated to the Lambda function.
        timeout (int): The maximum execution duration, in seconds, for the Lambda function.
        description (str): A description of the Lambda function.

    Returns:
        aws_cdk.aws_lambda.Function: The newly created Lambda function.

    """

    # Assuming 'construct' is an instance of the Construct class
    stack = aws_cdk.Stack.of(construct)
    stack_name = stack.stack_name

    if description == "":
        description = f"{stack_name.replace('-', '').title()} function for {lambda_fn_name.title()}"

    function_name = f"{stack_name}-{lambda_fn_name.title().replace('_', '')}"

    fn = lambda_.Function(
        construct,
        f"fn_{lambda_fn_name.title().replace('_', '')}",
        runtime=lambda_.Runtime.PYTHON_3_10,
        allow_public_subnet=True,
        code=lambda_.Code.from_asset(
            os.path.join(os.path.abspath(__file__), f"../../src/lambdas/{lambda_fn_name}")
        ),
        handler="handler.lambda_handler",
        function_name=function_name,
        memory_size=memory_size,
        retry_attempts=0,
        timeout=Duration.seconds(timeout),
        environment=envs,
        log_retention=logs.RetentionDays.ONE_MONTH,
        description=description,
        vpc=vpc,
        ephemeral_storage_size=kwargs.get(
            "ephemeral_storage_size", aws_cdk.Size.mebibytes(512)
        ),
        role=iam.Role(
            construct,
            f"{stack_name.replace('-', '').title()}{lambda_fn_name.title().replace('_', '')}Role",
            assumed_by=iam.ServicePrincipal("lambda.amazonaws.com"),
            description=f"Role for lambda {stack.artifact_id} {lambda_fn_name.title().replace('_', '')}",
            managed_policies=[
                iam.ManagedPolicy(
                    construct,
                    f"{lambda_fn_name.title().replace('_', '')}Policy",
                    statements=[
                        iam.PolicyStatement(
                            actions=["logs:CreateLogGroup"],
                            resources=[
                                f"arn:aws:logs:{aws_cdk.Aws.REGION}:{aws_cdk.Aws.ACCOUNT_ID}:"
                                + f"log-group:/aws/lambda/{function_name}"
                            ],
                        ),
                        iam.PolicyStatement(
                            actions=["logs:CreateLogStream", "logs:PutLogEvents"],
                            resources=[
                                f"arn:aws:logs:{aws_cdk.Aws.REGION}:{aws_cdk.Aws.ACCOUNT_ID}:"
                                + f"log-group:/aws/lambda/{function_name}",
                                f"arn:aws:logs:{aws_cdk.Aws.REGION}:{aws_cdk.Aws.ACCOUNT_ID}:"
                                + f"log-group:/aws/lambda/{function_name}:*",
                            ],
                        ),
                        iam.PolicyStatement(
                            actions=[
                                "ec2:CreateNetworkInterface",
                                "ec2:DescribeNetworkInterfaces",
                                "ec2:DescribeSubnets",
                                "ec2:DeleteNetworkInterface",
                                "ec2:AssignPrivateIpAddresses",
                                "ec2:UnassignPrivateIpAddresses",
                            ],
                            resources=[
                                "*",
                            ],
                        ),
                    ],
                )
            ],
        ),
    )

    return fn
