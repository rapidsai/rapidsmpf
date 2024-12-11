/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <rapidsmp/buffer.hpp>

namespace rapidsmp {


std::unique_ptr<Buffer> Buffer::copy_to_device() const {
    std::unique_ptr<rmm::device_buffer> ret;
    if (mem_type == MemType::device) {
        ret = std::make_unique<rmm::device_buffer>(
            device()->data(), device()->size(), stream, mr
        );
    } else {
        ret = std::make_unique<rmm::device_buffer>(
            host()->data(), host()->size(), stream, mr
        );
    }
    return std::make_unique<Buffer>(std::move(ret), stream, mr);
}

std::unique_ptr<Buffer> Buffer::copy_to_host() const {
    std::unique_ptr<std::vector<uint8_t>> ret;
    if (mem_type == MemType::host) {
        ret = std::make_unique<std::vector<uint8_t>>(*host());
    } else {
        ret = std::make_unique<std::vector<uint8_t>>(device()->size());
        RMM_CUDA_TRY(cudaMemcpy(
            ret->data(), device()->data(), device()->size(), cudaMemcpyDeviceToHost
        ));  // TODO: make async
    }
    return std::make_unique<Buffer>(std::move(ret), stream, mr);
}

std::unique_ptr<Buffer> Buffer::move(std::unique_ptr<Buffer>&& buffer, MemType target) {
    if (target != buffer->mem_type) {
        switch (buffer->mem_type) {
        case MemType::host:
            return buffer->copy_to_device();
        case MemType::device:
            return buffer->copy_to_host();
        }
    }
    return std::move(buffer);
}


}  // namespace rapidsmp
