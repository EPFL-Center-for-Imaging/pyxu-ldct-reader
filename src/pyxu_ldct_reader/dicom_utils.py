import sys

import pydicom as dicom
from pydicom.datadict import DicomDictionary, keyword_dict

from pyxu_ldct_reader.mayo_dicom_dict import new_dict_items

# Update the DICOM dictionary with the extra Mayo tags
DicomDictionary.update(new_dict_items)
# Update the reverse mapping from name to tag
keyword_dict.update({val[4]: tag for tag, val in new_dict_items.items()})


def _read_dicom_file(file_path, file_name):
    try:
        return dicom.read_file(file_path)
    except Exception as e:
        print(f"Corrupted file: {file_name}", file=sys.stderr)
        raise e
