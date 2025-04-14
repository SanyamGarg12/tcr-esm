# TCR-ESM Predictor

A web-based tool for predicting TCR-peptide-MHC interactions using ESM-based models.

## Project Structure

```
tcr-esm/
├── backend/           # Backend code and models
│   ├── cgi-bin/      # CGI scripts for handling predictions
│   └── models/       # Trained model files
├── frontend/         # Frontend web interface
│   ├── index.html    # Main web page
│   ├── style.css     # Styling
│   ├── script.js     # Client-side functionality
│   └── uploads/      # Directory for uploaded files
├── docs/             # Documentation
└── sample_data/      # Sample input data
```

## Features

- Multiple prediction tasks support:
  - TCRα-TCRβ-Peptide-MHC
  - TCRα-TCRβ-Peptide
  - TCRα-Peptide-MHC
  - TCRβ-Peptide-MHC
  - TCRα-Peptide
  - TCRβ-Peptide
- Compatible with MCPAS and VDJDB databases
- User-friendly web interface
- Sample data available for testing

## Installation

### Prerequisites

- XAMPP (Apache + PHP)
- Python 3.x
- Required Python packages:
  - tensorflow
  - numpy
  - keras

### Setup Instructions

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/tcr-esm.git
   ```

2. Copy the files to your XAMPP installation:
   - Copy `frontend/` contents to `xampp/htdocs/tcr-esm/`
   - Copy `backend/cgi-bin/` contents to `xampp/cgi-bin/`
   - Copy `backend/models/` to `xampp/cgi-bin/models/`

3. Install required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Configure Apache:
   - Enable CGI module
   - Set proper permissions for uploads directory

Detailed installation instructions can be found in [INSTALLATION.md](docs/INSTALLATION.md).

## Usage

1. Start XAMPP and ensure Apache is running
2. Access the web interface at `http://localhost/tcr-esm`
3. Select your prediction task and database
4. Upload the required embedding files
5. Submit for prediction

For detailed usage instructions, see [USAGE.md](docs/USAGE.md).

## File Requirements

- All input files must be in .npy format (NumPy array format)
- Files should contain embeddings generated using the ESM model
- Required files depend on the selected task:
  - TCRα Embedding: For tasks involving alpha chain
  - TCRβ Embedding: For tasks involving beta chain
  - Peptide Embedding: Required for all tasks
  - MHC Embedding: For tasks involving MHC

## Contributing

Contributions are welcome! Please read our [contributing guidelines](CONTRIBUTING.md) before submitting pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or support, please contact:
- Email: sanyam22448@iiitd.ac.in
- Institute: IIIT Delhi 