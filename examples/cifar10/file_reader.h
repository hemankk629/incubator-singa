/**
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef SINGA_FILE_READER_H_
#define SINGA_FILE_READER_H_

#include <string>
#include <cstring>
#include <memory>
#include <fstream>

#include "singa/core/tensor.h"
#include "singa/utils/logging.h"

namespace singa {

class FileReader {
public:
	void OpenForRead(std::string file_name);

	void OpenForWrite(std::string file_name);

	void Close();

	bool Read(std::string* key, uint8_t* bytes, size_t* size);

	std::vector<std::pair<std::string, Tensor>> Read();

	void Write(std::string key, uint8_t* bytes, size_t size);

	void Write(std::string key, Tensor tensor);

private:
	std::ofstream fout_;
	std::ifstream fin_;
	bool isOpenForWrite_ = false;
	bool isOpenForRead_ = false;
};

} // namespace singa

#endif // SINGA_FILE_READER_H_
