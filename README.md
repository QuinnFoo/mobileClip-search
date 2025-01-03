#mobileClip-Search: Image Search Application

An image search application based on Apple's mobileClip-Search model, supporting both text-to-image and image-to-image search capabilities.

## Features

- Text-to-image search
- Image-to-image similarity search
- Support for both local and AWS S3 storage
- Docker containerization
- AWS EC2 cloud deployment
- Real-time image processing and search
- Efficient vector indexing with FAISS

## Tech Stack

- **Backend**: Flask
- **Deep Learning**: PyTorch
- **Model**: Apple mobileClip-Search
- **Vector Search**: FAISS
- **Deployment**: Docker, AWS EC2
- **Storage**: Local filesystem / AWS S3
- **Web Server**: Gunicorn

## Quick Start

### Local Development

1. Clone the repository and install dependencies:
```bash
git clone [your-repository-url]
cd mobileClip-Search
pip install -r requirements.txt
```

2. Download the model:
```bash
mkdir -p checkpoints
# Download mobileclip_s0.pt to the checkpoints directory
```

3. Configure environment variables:
```bash
cp .env.example .env
# Edit .env file with your configuration
```

4. Run the application:
```bash
python app.py
```

### Docker Deployment

1. Build and run with Docker:
```bash
docker-compose up --build
```

### AWS Deployment

1. Prepare AWS environment:
   - Create an EC2 instance
   - Configure security group (open port 80)
   - Create S3 bucket (if using S3 mode)

2. Configure deployment:
   - Copy SSH key to ~/.ssh/
   - Configure .env file

3. Deploy:
```bash
bash deploy/deploy.sh
```

## Project Structure

```
mobileClip-Search/
├── app.py                 # Main application file
├── requirements.txt       # Python dependencies
├── templates/            # HTML templates
│   └── index.html        # Main page
├── checkpoints/          # Model directory
│   └── mobileclip_s0.pt  # mobileClip-Search model
├── images/              # Image library directory
├── cache/               # Cache directory
├── deploy/              # Deployment files
│   ├── deploy.sh        # Deployment script
│   ├── Dockerfile       # Docker configuration
│   └── docker-compose.yml # Docker Compose configuration
└── scripts/             # Utility scripts
    └── logs.sh          # Log viewing script
```

## Environment Configuration

Configure the following variables in your `.env` file:

```env
# AWS Credentials
AWS_ACCESS_KEY_ID=your_access_key_id
AWS_SECRET_ACCESS_KEY=your_secret_access_key
AWS_DEFAULT_REGION=ap-southeast-2
S3_BUCKET=your-bucket-name

# EC2 Configuration
EC2_IP=your.ec2.ip.address
EC2_KEY=path/to/your/key.pem

# Application Configuration
STORAGE_TYPE=s3  # or 'local'
FLASK_ENV=production
```

## Usage Guide

1. **Text Search**:
   - Enter descriptive text in the search box
   - Click the "Search" button
   - View the most relevant images

2. **Image Search**:
   - Click the "Upload Image" button
   - Select an image file
   - View similar images

## Maintenance and Monitoring

### View Logs
```bash
# View real-time logs
bash scripts/logs.sh -f

# View last 100 lines
bash scripts/logs.sh -n 100

# View logs with timestamps
bash scripts/logs.sh -t
```

### Update Deployment
```bash
# Update application
bash deploy/deploy.sh
```