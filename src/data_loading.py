"""
Utilities for loading clinical data and MRI images for dementia ML workflows.
"""
import pandas as pd
import os

def load_clinical_data(filepath, **kwargs):
    """
    Load clinical data from CSV or Excel files into a pandas DataFrame.

    Args:
        filepath (str): Path to the clinical data file (CSV or Excel).
        **kwargs: Additional arguments passed to pandas read functions.
    Returns:
        pd.DataFrame: Loaded clinical data.
    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If file extension is not supported.
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    ext = os.path.splitext(filepath)[-1].lower()
    if ext in ['.csv']:
        return pd.read_csv(filepath, **kwargs)
    elif ext in ['.xls', '.xlsx']:
        return pd.read_excel(filepath, **kwargs)
    else:
        raise ValueError("Unsupported file extension for clinical data: {}".format(ext))

# MRI image loading utilities
try:
    import nibabel as nib
except ImportError:
    nib = None
try:
    import pydicom
except ImportError:
    pydicom = None

def load_mri_image(filepath, type_hint=None):
    """
    Load an MRI image from NIfTI (.nii/.nii.gz) or DICOM (".dcm") file.

    Args:
        filepath (str): File path to the MRI image.
        type_hint (str, optional): If set to 'nifti' or 'dicom', enforces which loader to use.
    Returns:
        data: Image data array or DICOM object.
    Raises:
        ImportError: If required libraries are not installed.
        FileNotFoundError: If the file does not exist.
        ValueError: If file type is unsupported.
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    ext = os.path.splitext(filepath)[-1].lower()

    if type_hint == 'nifti' or ext in ['.nii', '.nii.gz']:
        if nib is None:
            raise ImportError("nibabel library is required for loading NIfTI files.")
        img = nib.load(filepath)
        return img.get_fdata()
    elif type_hint == 'dicom' or ext in ['.dcm']:
        if pydicom is None:
            raise ImportError("pydicom library is required for loading DICOM files.")
        return pydicom.dcmread(filepath)
    else:
        raise ValueError(f"Unsupported or undetected MRI file type: {filepath}")
