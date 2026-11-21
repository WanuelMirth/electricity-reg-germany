output "setup_notebook_url" {
  description = "Open the audit notebook and select Serverless to run interactively."
  value       = "${trimsuffix(var.databricks_host, "/")}/#workspace${trimsuffix(var.databricks_project_path, "/")}/notebooks/01_raw_data_audit.ipynb"
}

output "setup_job_url" {
  description = "Open the serverless audit job; click Run now to execute it."
  value       = databricks_job.setup_check.url
}

output "setup_job_id" {
  description = "Job ID used by GitHub Actions to trigger the serverless run."
  value       = databricks_job.setup_check.id
}
