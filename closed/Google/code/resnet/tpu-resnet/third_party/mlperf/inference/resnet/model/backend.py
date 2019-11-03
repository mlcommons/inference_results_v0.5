# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Abstract backend class definition."""


class Backend(object):
  """Abstract backend class definition."""

  def __init__(self):
    self.inputs = []
    self.outputs = []

  def version(self):
    raise NotImplementedError("Backend:version")

  def name(self):
    raise NotImplementedError("Backend:name")

  def load(self, model_path, inputs=None, outputs=None):
    raise NotImplementedError("Backend:load")

  def predict(self, feed):
    raise NotImplementedError("Backend:predict")
