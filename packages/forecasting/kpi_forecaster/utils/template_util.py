# python3
# coding=utf-8
# Copyright 2021 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utility functions for the SQL templating."""

import os
from typing import Any, Mapping

import jinja2


def render_template(template_file: str,
                    template_params: Mapping[str, Any]) -> str:
  """Render the Jinja template_file using template_params.

  Args:
    template_file: String name of the file containing the template. The file
       name can be absolute (e.g. if it starts with '/'), or relative to the
       current module path.
    template_params: Mapping from str param name to param value.

  Returns:
    template_file string with param names substituted by template_params.
  """
  return jinja2.Environment(
      loader=jinja2.FileSystemLoader(
          os.path.join(os.path.dirname(__file__))),
      keep_trailing_newline=True,
      trim_blocks=True,
      lstrip_blocks=True).get_template(template_file).render(template_params)
