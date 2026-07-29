# 🧪 Biofilm Analyzer

Biofilm Analyzer is an interactive web application for semantic segmentation and morphometric analysis of scanning electron microscopy (SEM) images of bacterial biofilms.

The application combines deep learning-based segmentation with semantic filtering of connected components, allowing researchers to visualize segmentation results, interactively filter detected objects, calculate morphometric statistics, and export annotations in CVAT format.

---

## Features

- 🔬 Semantic segmentation of SEM images
- 🧫 Support for multiple bacterial morphologies
- ⚙️ Interactive filtering by:
  - object area
  - eccentricity
- 📊 Automatic morphometric statistics
- 📦 Export of:
  - segmentation overlays
  - Excel reports
  - CVAT-compatible annotations
- 📏 Automatic and manual image scale detection
- 🌐 Interactive Streamlit interface

---

## Installation

Clone the repository

```bash
git clone https://github.com/<username>/biofilm-analyzer.git
cd biofilm-analyzer
```

Create a virtual environment

```bash
python -m venv .venv
```

Activate it

Windows

```bash
.venv\Scripts\activate
```

Linux

```bash
source .venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## Usage

Run the application

```bash
streamlit run app.py
```

The application will become available at

```
http://localhost:8501
```

---

## Workflow

1. Upload an SEM image.
2. Select the bacterial morphology model.
3. Run segmentation.
4. Adjust morphometric filters.
5. Review statistics.
6. Export the processed results or CVAT annotations.

---

## Repository structure

```
BiofilmAnalyzer/
│
├── app.py
├── core/
├── models/
├── segmentation/
├── styles/
├── utils/
├── requirements.txt
└── README.md
```

---

## Citation

If you use this software in your research, please cite:

> Pavlova, V. S., Kurbakov, M. Yu., Kopylov, A. V., & Seredin, O. S.
> **Segmentation of Microscopic Images of Pseudomonas Biofilms Using Deep Learning and Semantic Filtering.**
> *Pattern Recognition and Image Analysis*, 2026, **36**(2), 777–792.
> https://doi.org/10.1134/S1054661826700586

BibTeX

```bibtex
@article{Pavlova2026Biofilm,
  author  = {V. S. Pavlova and M. Yu. Kurbakov and A. V. Kopylov and O. S. Seredin},
  title   = {Segmentation of Microscopic Images of Pseudomonas Biofilms Using Deep Learning and Semantic Filtering},
  journal = {Pattern Recognition and Image Analysis},
  year    = {2026},
  volume  = {36},
  number  = {2},
  pages   = {777--792},
  doi     = {10.1134/S1054661826700586}
}
```

---

## License

This project is distributed under the **PolyForm Strict License 1.0.0**.

See the `LICENSE` file for details.
