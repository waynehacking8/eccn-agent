from projen.awscdk import AwsCdkPythonApp

project = AwsCdkPythonApp(
    author_email="gcr_genai@amazon.com",
    author_name="GCR-AIFL",
    cdk_version="2.1.0",
    github=False,
    module_name="src",
    name="src",
    poetry=False,
    python_exec="python3",
    version="0.1.0",
    dev_deps=["black", "boto3", "opensearch-py", "requests", "pymupdf4llm"],
)

project.pre_compile_task.exec(
    "cd src/lambdas/mcp_server/ && tsc --project tsconfig.json"
)

project.pre_compile_task.exec(
    "cd src/lambdas/eccn_client/ && tsc --project tsconfig.json"
)

project.pre_compile_task.exec("./create_layer.sh")


project.synth()
