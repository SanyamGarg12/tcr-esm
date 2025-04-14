# Installation Guide

This guide will help you set up the TCR-ESM Predictor on your system.

## System Requirements

- Windows/Linux/MacOS
- XAMPP (with Apache and PHP)
- Python 3.x
- Git (optional, for cloning the repository)

## Step 1: Install XAMPP

1. Download XAMPP from [https://www.apachefriends.org/](https://www.apachefriends.org/)
2. Install XAMPP following the installation wizard
3. Start Apache from the XAMPP Control Panel

## Step 2: Clone or Download the Repository

### Option 1: Using Git
```bash
git clone https://github.com/yourusername/tcr-esm.git
```

### Option 2: Manual Download
1. Download the repository as a ZIP file
2. Extract the contents to a temporary location

## Step 3: Set Up the Project Structure

1. Create the following directories in your XAMPP installation:
   ```
   xampp/htdocs/tcr-esm/
   xampp/cgi-bin/
   ```

2. Copy the files:
   - Copy `frontend/` contents to `xampp/htdocs/tcr-esm/`
   - Copy `backend/cgi-bin/` contents to `xampp/cgi-bin/`
   - Copy `backend/models/` to `xampp/cgi-bin/models/`

## Step 4: Install Python Dependencies

1. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install required packages:
   ```bash
   pip install tensorflow numpy keras
   ```

## Step 5: Configure Apache

1. Enable CGI module:
   - Open `xampp/apache/conf/httpd.conf`
   - Uncomment the line: `LoadModule cgi_module modules/mod_cgi.so`

2. Configure CGI directory:
   - Add or modify the following in `httpd.conf`:
   ```apache
   <Directory "/path/to/xampp/cgi-bin">
       Options +ExecCGI
       AddHandler cgi-script .py
       Require all granted
   </Directory>
   ```

3. Set proper permissions:
   ```bash
   chmod 755 /path/to/xampp/cgi-bin/*.py
   chmod 777 /path/to/xampp/htdocs/tcr-esm/uploads
   ```

## Step 6: Test the Installation

1. Start XAMPP and ensure Apache is running
2. Access the web interface at `http://localhost/tcr-esm`
3. Try uploading sample data to verify the installation

## Troubleshooting

### Common Issues

1. **CGI Scripts Not Executing**
   - Verify CGI module is enabled in Apache
   - Check file permissions
   - Ensure Python path is correct in CGI scripts

2. **File Upload Issues**
   - Check uploads directory permissions
   - Verify PHP upload settings in php.ini

3. **Model Loading Errors**
   - Ensure model files are in the correct location
   - Check Python dependencies are installed

### Getting Help

If you encounter any issues:
1. Check the Apache error logs
2. Review the Python script output
3. Contact the maintainers for support

## Next Steps

After installation, proceed to [USAGE.md](USAGE.md) for instructions on using the predictor. 