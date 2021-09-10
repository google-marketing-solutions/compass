# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Propensity model custom utility functions."""

import logging
import pathlib
import sys
from typing import List, Union

from matplotlib.backends import backend_pdf
import numpy as np

logging.basicConfig(
    format='%(levelname)s: %(message)s', level=logging.INFO, stream=sys.stdout)

_VISUALIZATION_OUTPUT_DIR = 'visualization_outputs'


def create_folder(folder_name: str) -> pathlib.Path:
  path = pathlib.Path(folder_name)
  try:
    path.mkdir(parents=True, exist_ok=False)
  except FileExistsError:
    logging.warning('Folder "%s" already exists', folder_name)
  else:
    logging.info('Created "%s".', folder_name)
  return path.absolute()


def save_to_pdf(filename: str, plots: Union[np.ndarray,
                                            List[np.ndarray]]) -> None:
  """Saves pyplot axes into a PDF file.

  Args:
    filename: PDF filename to save plots.
    plots: List of pyplot axes to save to PDF.

  Raises:
    TypeError if plots are not in List[np.ndarray] or np.ndarray type.
  """
  abs_path = create_folder(_VISUALIZATION_OUTPUT_DIR)
  pdf_path = pathlib.Path(abs_path, filename)
  logging.info('Creating PDF file in "%s"', pdf_path)
  pdf = backend_pdf.PdfPages(pdf_path)
  if all(isinstance(plot, np.ndarray) for plot in plots):
    for plot in plots:
      pdf.savefig(plot[0].get_figure())
  elif isinstance(plots, np.ndarray):
    pdf.savefig(plots[0].get_figure())
  else:
    raise TypeError('Plots list is not supported. Provide either '
                    'List[np.ndarray] or np.ndarray containing pyplot axes.')
  pdf.close()
