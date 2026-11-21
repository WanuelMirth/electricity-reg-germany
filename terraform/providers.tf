terraform {
  required_version = ">= 1.7.0, < 2.0.0"

  required_providers {
    databricks = {
      source  = "databricks/databricks"
      version = "~> 1.114.0"
    }
  }
}

# Local use defaults to `az login`; GitHub Actions uses a Databricks token.
provider "databricks" {
  host      = var.databricks_host
  auth_type = var.databricks_auth_type
}
