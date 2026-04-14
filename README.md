# Gluemancy — Spark Ingest Demo

A PySpark data-ingestion demo that generates ~10 K mock records covering nearly every Spark SQL type, runs locally via `pyspark`, and deploys to **AWS Glue 4.0** for cloud execution.

---

## Project Structure

```
.
├── .env                          # Local env vars (DO NOT commit — contains secrets)
├── Dockerfile
├── justfile                      # Task runner (requires `just`)
├── pyproject.toml
├── src/
│   ├── spark_ingest_demo.py      # Local PySpark job (writes to /workspace/output/)
│   └── spark_ingest_demo_glue.py # AWS Glue job (reads/writes S3)
├── logs/
└── output/
    └── spark_demo_csv/           # Local CSV output
```

---

## Prerequisites

| Tool | Purpose |
|---|---|
| Python ≥ 3.12 | Runtime |
| [uv](https://docs.astral.sh/uv/) | Dependency management |
| [just](https://github.com/casey/just) | Task runner |
| [AWS CLI v2](https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2.html) | Glue / S3 operations |

---

## Local Setup

```bash
# Install dependencies
just sync

# Run the local PySpark job
python src/spark_ingest_demo.py
# Output: output/spark_demo_csv/part-*.csv
```

### Environment Variables

Copy `.env.example` and rename to `.env`, then fill in your values (never commit real credentials):

```
S3_BUCKET=<your-s3-bucket>
AWS_DEFAULT_REGION=us-east-1
AWS_ACCESS_KEY_ID=<your-access-key>
AWS_SECRET_ACCESS_KEY=<your-secret-key>
```

---

## AWS Glue Deployment

### 1. Configure AWS CLI

```bash
aws configure
# Enter your AWS Access Key ID, Secret, region (us-east-1), and output format (json)
```

---

### 2. IAM Setup

#### 2a. Create the `GlueJobDeploy` policy

In the AWS Console (or via CLI), create a managed policy named **`GlueJobDeploy`** with the following document:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "glue:CreateJob",
        "glue:UpdateJob",
        "glue:DeleteJob",
        "glue:GetJob",
        "glue:GetJobs",
        "glue:StartJobRun",
        "glue:GetJobRun",
        "glue:GetJobRuns",
        "glue:BatchStopJobRun"
      ],
      "Resource": "arn:aws:glue:us-east-1:977237815226:job/*"
    },
    {
      "Effect": "Allow",
      "Action": "iam:PassRole",
      "Resource": "arn:aws:iam::977237815226:role/AWSGlueRole-SparkDemo"
    }
  ]
}
```

#### 2b. Attach policies to the `ascendio-dev` IAM user

Attach all three of the following to the **`ascendio-dev`** user:

| Policy | Type |
|---|---|
| `AWSGlueConsoleFullAccess` | AWS Managed |
| `GlueJobDeploy` | Customer Managed (created above) |
| `AWSGlueServiceRole` | AWS Managed |

---

### 3. Upload the Glue script to S3

```bash
aws s3 cp src/spark_ingest_demo_glue.py \
  s3://ascendio-s3-bucket/glue-scripts/spark_ingest_demo_glue.py
```

Create the output prefix (S3 "folder"):

```bash
aws s3api put-object \
  --bucket ascendio-s3-bucket \
  --key output/spark_demo_csv/
```

---

### 4. Create the Glue Job

```bash
aws glue create-job \
  --name spark-ingest-demo \
  --role AWSGlueRole-SparkDemo \
  --command '{
    "Name": "glueetl",
    "ScriptLocation": "s3://ascendio-s3-bucket/glue-scripts/spark_ingest_demo_glue.py",
    "PythonVersion": "3"
  }' \
  --glue-version "4.0" \
  --worker-type G.1X \
  --number-of-workers 2 \
  --default-arguments '{
    "--output_path": "s3://ascendio-s3-bucket/output/spark_demo_csv/",
    "--num_records": "10000",
    "--TempDir": "s3://ascendio-s3-bucket/glue-tmp/",
    "--enable-continuous-cloudwatch-log": "true"
  }'
```

---

### 5. Run the Glue Job

```bash
aws glue start-job-run --job-name spark-ingest-demo
```

The command returns a **Run ID** (`jr_...`). Use it to monitor progress:

```bash
aws glue get-job-run \
  --job-name spark-ingest-demo \
  --run-id <run-id> \
  --query 'JobRun.{State:JobRunState,Error:ErrorMessage}'
```

**Expected states:** `STARTING` → `RUNNING` → `SUCCEEDED`

Output CSV parts are written to:
```
s3://ascendio-s3-bucket/output/spark_demo_csv/
```

---

## Available `just` Commands

```
just sync          # Install / sync all dependency groups
just add <pkg>     # Add a runtime dependency
just add-dev <pkg> # Add a dev-only dependency
just remove <pkg>  # Remove a dependency
just update        # Upgrade all locked dependencies
just deps          # Show the resolved dependency tree
just export        # Export pinned requirements.txt files
```
