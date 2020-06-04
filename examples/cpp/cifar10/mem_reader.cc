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

#include "mem_reader.h"
#include "singa/utils/logging.h"

namespace singa {

MemReader::MemReader(char* src, int size) : buf_(src), bufsize_(size), offset_(0) {};

bool MemReader::Read(std::string* key, std::string* value) {
  char magic[4];
  int smagic = sizeof(magic);
  memcpy(magic, buf_ + offset_, smagic);
  offset_ += smagic;
  LOG(INFO) << "Offset " << offset_ << " size " << bufsize_;
  if (offset_ >= bufsize_) {
	  LOG(INFO) << "End of parameters";
	  return false;
  }
  if (magic[0] == kMagicWord[0] && magic[1] == kMagicWord[1]) {
    if (magic[2] != 0 && magic[2] != 1) {
      LOG(INFO) << "File format error: magic word does not match!" << magic;
      return false;
    }
    if (magic[2] == 1)
      if (!ReadField(key)) return false;
    if (!ReadField(value)) return false;
  }
  else {
    LOG(INFO) << "File format error: magic word does not match!" << magic;
    return false;
  }
  return true;
}

void MemReader::SeekToFirst() {
  offset_ = 0;
}

bool MemReader::ReadField(std::string* content) {
  content->clear();
  int ssize = sizeof(size_t);
  int len = *reinterpret_cast<int*>(buf_ + offset_);
  offset_ += ssize;
  content->reserve(len);
  content->insert(0, buf_ + offset_, len);
  //for (int i = 0; i < len; ++i) content->push_back(buf_[offset_ + i]);
  offset_ += len;
  return true;
}

}  // namespace singa
