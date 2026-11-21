# Stop managing the old upload without deleting any existing notebook.
removed {
  from = databricks_notebook.setup_check
  lifecycle {
    destroy = false
  }
}

resource "databricks_job" "setup_check" {
  name                = "electricity-demand-raw-data-audit"
  max_concurrent_runs = 1
  timeout_seconds     = 600

  task {
    task_key    = "raw_data_audit"
    max_retries = 0

    # No cluster configuration: runs on serverless compute.
    notebook_task {
      notebook_path = "${trimsuffix(var.databricks_project_path, "/")}/notebooks/01_raw_data_audit.ipynb"
      source        = "WORKSPACE"
    }
  }
}
