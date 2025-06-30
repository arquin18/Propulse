# Propulse Infrastructure

This directory contains all infrastructure-as-code (IaC) configurations and deployment scripts for the Propulse system on Google Cloud Platform (GCP).

## 📁 Directory Structure

```
infra/
├── gcp/             # GCP-specific configurations
│   ├── backend/     # Backend service configs
│   └── frontend/    # Frontend service configs
└── terraform/       # Terraform configurations
    ├── modules/     # Reusable Terraform modules
    ├── prod/        # Production environment
    └── staging/     # Staging environment
```

## 🌩️ Cloud Architecture

### Services Used
- Cloud Run: Container hosting
- Cloud Storage: File storage
- Secret Manager: Secrets management
- Cloud Logging: Centralized logging
- Cloud Monitoring: System monitoring

### Network Architecture
- VPC configuration
- Load balancing
- Security policies
- Service mesh

## 🚀 Deployment

### Prerequisites
1. Install required tools:
   ```bash
   # Install Terraform
   brew install terraform
   
   # Install Google Cloud SDK
   brew install google-cloud-sdk
   
   # Authenticate with GCP
   gcloud auth login
   gcloud config set project YOUR_PROJECT_ID
   ```

2. Configure environment:
   ```bash
   # Set up application default credentials
   gcloud auth application-default login
   
   # Enable required APIs
   gcloud services enable \
     run.googleapis.com \
     secretmanager.googleapis.com \
     storage.googleapis.com
   ```

### Deployment Steps

1. **Initialize Terraform**
   ```bash
   cd terraform/prod
   terraform init
   ```

2. **Plan Changes**
   ```bash
   terraform plan -out=tfplan
   ```

3. **Apply Changes**
   ```bash
   terraform apply tfplan
   ```

## 🔧 Configuration

### Backend Service
```hcl
# Example Cloud Run configuration
resource "google_cloud_run_service" "backend" {
  name     = "propulse-backend"
  location = var.region

  template {
    spec {
      containers {
        image = "gcr.io/${var.project_id}/backend:latest"
        
        resources {
          limits = {
            cpu    = "1000m"
            memory = "512Mi"
          }
        }
      }
    }
  }
}
```

### Frontend Service
```hcl
# Example Cloud Run configuration
resource "google_cloud_run_service" "frontend" {
  name     = "propulse-frontend"
  location = var.region

  template {
    spec {
      containers {
        image = "gcr.io/${var.project_id}/frontend:latest"
        
        resources {
          limits = {
            cpu    = "500m"
            memory = "256Mi"
          }
        }
      }
    }
  }
}
```

## 🔒 Security

### IAM Configuration
- Service accounts
- Role bindings
- Custom roles

### Network Security
- VPC Service Controls
- Cloud Armor policies
- SSL certificates

## 📊 Monitoring

### Metrics
- CPU usage
- Memory usage
- Request latency
- Error rates

### Alerts
- Service health
- Resource utilization
- Cost thresholds
- Security events

## 💰 Cost Management

### Resource Optimization
- Autoscaling policies
- Resource limits
- Scheduled scaling

### Cost Controls
- Budget alerts
- Resource quotas
- Usage monitoring

## 🔄 CI/CD Integration

### GitHub Actions
- Automated deployments
- Infrastructure validation
- Security scanning

### Deployment Strategies
- Blue/green deployment
- Canary releases
- Rollback procedures

## 🐛 Troubleshooting

### Common Issues
1. Permission errors
2. Resource limits
3. Network connectivity
4. Service dependencies

### Debug Tools
- Cloud Trace
- Cloud Debugger
- Cloud Profiler
- Error Reporting

## 📚 Additional Resources

- [GCP Documentation](https://cloud.google.com/docs)
- [Terraform Documentation](https://www.terraform.io/docs)
- [Cloud Run Documentation](https://cloud.google.com/run/docs)

## 🤝 Contributing

1. Follow IaC best practices
2. Update documentation
3. Test changes in staging
4. Create pull request 