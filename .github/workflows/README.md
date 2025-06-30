# Propulse CI/CD Workflows

This directory contains GitHub Actions workflows for continuous integration and deployment of the Propulse system.

## 📁 Workflow Files

```
workflows/
├── backend-ci.yml      # Backend CI pipeline
├── frontend-ci.yml     # Frontend CI pipeline
├── backend-deploy.yml  # Backend deployment
├── frontend-deploy.yml # Frontend deployment
└── infra-deploy.yml   # Infrastructure deployment
```

## 🔄 CI Workflows

### Backend CI (`backend-ci.yml`)
```yaml
name: Backend CI

on:
  push:
    paths:
      - 'backend/**'
      - 'shared/**'
  pull_request:
    paths:
      - 'backend/**'
      - 'shared/**'

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - name: Run Tests
      - name: Run Linting
      - name: Check Types
      - name: Build Container
```

### Frontend CI (`frontend-ci.yml`)
```yaml
name: Frontend CI

on:
  push:
    paths:
      - 'frontend/**'
      - 'shared/**'
  pull_request:
    paths:
      - 'frontend/**'
      - 'shared/**'

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - name: Run Tests
      - name: Run Linting
      - name: Build Container
```

## 🚀 Deployment Workflows

### Backend Deployment (`backend-deploy.yml`)
- Triggers on release tags
- Builds and pushes container
- Deploys to Cloud Run
- Runs integration tests
- Handles rollbacks

### Frontend Deployment (`frontend-deploy.yml`)
- Triggers on release tags
- Builds and pushes container
- Deploys to Cloud Run
- Runs smoke tests
- Updates DNS

### Infrastructure Deployment (`infra-deploy.yml`)
- Triggers on infrastructure changes
- Validates Terraform
- Plans changes
- Applies changes
- Updates documentation

## 🔒 Security

### Secrets Management
- GCP credentials
- API keys
- Environment variables
- Docker registry credentials

### Access Control
- Branch protection rules
- Required reviewers
- Environment protection rules
- Deployment approvals

## 📊 Monitoring

### Job Metrics
- Build time
- Test coverage
- Success rate
- Resource usage

### Notifications
- Slack alerts
- Email notifications
- GitHub notifications
- Status checks

## 🔧 Configuration

### Environment Setup
```yaml
env:
  PROJECT_ID: ${{ secrets.GCP_PROJECT_ID }}
  REGION: us-central1
  REGISTRY: gcr.io
```

### Container Build
```yaml
- name: Build and Push
  uses: docker/build-push-action@v2
  with:
    context: .
    push: true
    tags: ${{ env.REGISTRY }}/${{ env.PROJECT_ID }}/app:${{ github.sha }}
```

## 🐛 Troubleshooting

### Common Issues
1. Failed builds
   - Check logs
   - Verify dependencies
   - Check resource limits

2. Failed deployments
   - Verify credentials
   - Check service health
   - Review configuration

3. Infrastructure issues
   - Check Terraform state
   - Verify permissions
   - Review change plan

## 📚 Best Practices

1. **Workflow Organization**
   - Clear naming conventions
   - Modular job structure
   - Reusable actions

2. **Security**
   - Secret scanning
   - SAST/DAST
   - Dependency scanning

3. **Performance**
   - Caching strategies
   - Parallel jobs
   - Resource optimization

## 🔄 Maintenance

### Regular Tasks
- Update action versions
- Review dependencies
- Clean up artifacts
- Update documentation

### Monitoring
- Check workflow analytics
- Review usage limits
- Monitor costs
- Track performance

## 📚 Additional Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Workflow Syntax](https://docs.github.com/en/actions/reference/workflow-syntax-for-github-actions)
- [GitHub Actions Marketplace](https://github.com/marketplace?type=actions)

## 🤝 Contributing

1. Test workflows locally
2. Update documentation
3. Follow security guidelines
4. Create pull request 