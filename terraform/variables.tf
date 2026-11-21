variable "databricks_host" {
  description = "Existing Azure Databricks workspace URL, without a notebook path or query string."
  type        = string

  validation {
    condition     = can(regex("^https://adb-[a-zA-Z0-9.-]+\\.azuredatabricks\\.net/?$", var.databricks_host))
    error_message = "Use the workspace URL, for example https://adb-1234567890123456.7.azuredatabricks.net."
  }
}

variable "databricks_auth_type" {
  description = "azure-cli locally, pat for GitHub Actions using DATABRICKS_TOKEN."
  type        = string
  default     = "azure-cli"

  validation {
    condition     = contains(["azure-cli", "pat"], var.databricks_auth_type)
    error_message = "Use azure-cli or pat."
  }
}

variable "databricks_project_path" {
  description = "Shared main Git folder path used by the audit job."
  type        = string
  default     = "/Shared/electricity-demand/branches/main"

  validation {
    condition     = startswith(var.databricks_project_path, "/")
    error_message = "Use the absolute workspace path copied from your Databricks Git folder."
  }
}
