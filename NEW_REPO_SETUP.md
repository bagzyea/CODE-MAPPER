# Setting Up New Repository for CODE-MAPPER

Follow these steps to create and push your code to a new repository.

## Step 1: Create New Repository on GitHub

1. Go to https://github.com
2. Click "New" or "+" in the top-right corner
3. Choose "New repository"
4. Fill in the details:
   - **Repository name**: `CODE-MAPPER` (or your preferred name)
   - **Description**: "Industry Classification Code Mapper - AI-powered ISIC/ISCO classification tool"
   - **Visibility**: Choose Public or Private
   - **DON'T** initialize with README, .gitignore, or license (we already have these)

## Step 2: Add New Remote Origin

Replace `YOUR_USERNAME` with your GitHub username and `YOUR_REPO_NAME` with your repository name:

```bash
# Add new remote origin
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git

# Verify remote is added
git remote -v
```

## Step 3: Prepare and Push Code

```bash
# Add all files to staging
git add .

# Commit changes
git commit -m "Initial commit: Industry Classification Code Mapper

- Streamlit web application for ISIC/ISCO classification
- Docker deployment support for Windows and Linux
- Fine-tuned model integration
- Production-ready nginx configuration
- Comprehensive deployment documentation"

# Push to new repository
git push -u origin master
```

## Step 4: Verify Push

1. Go to your new repository on GitHub
2. Verify all files are present
3. Check that the README.md displays correctly

## Alternative: Using GitHub CLI

If you have GitHub CLI installed:

```bash
# Create repository directly from command line
gh repo create CODE-MAPPER --public --source=. --remote=origin --push

# Or for private repository
gh repo create CODE-MAPPER --private --source=. --remote=origin --push
```

## After Repository Setup

Update these files if needed:
- Update repository URL in `DEPLOYMENT_GUIDE_PRODUCTION.md`
- Update clone URLs in documentation
- Add repository-specific badges to README.md

## Repository Structure

Your repository will contain:
```
CODE-MAPPER/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── Dockerfile                      # Docker configuration
├── docker-compose.yml             # Development Docker setup
├── docker-compose.prod.yml        # Production Docker setup
├── nginx.conf                      # Nginx reverse proxy config
├── deploy.sh                       # Linux deployment script
├── deploy-windows.ps1              # Windows deployment script
├── fine_tuned_classifier.py        # Fine-tuned model integration
├── README.md                       # Main documentation
├── DEPLOYMENT_GUIDE_PRODUCTION.md  # Detailed deployment guide
├── WINDOWS_DEPLOYMENT.md           # Windows-specific deployment
├── data/
│   ├── Localised ISIC.xlsx         # ISIC classification data
│   └── isco_index.xlsx             # ISCO classification data
└── outputs/                        # User-generated files directory
```

## Next Steps

After setting up the repository:
1. Deploy to your Windows server using the Windows deployment guide
2. Configure domain and SSL if needed
3. Set up monitoring and backup procedures