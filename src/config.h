#pragma once 
#include "rapidjson/document.h"
#include "rapidjson/filereadstream.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <string>
#include "rapidjson/prettywriter.h"
#include "rapidjson/stringbuffer.h"
using namespace rapidjson;

class Config {
public:
    Document doc;

    Config(const std::string& filePath) {
        loadConfig(filePath);
    }

    void loadConfig(const std::string& filePath) {
        std::ifstream file(filePath);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open config file: " + filePath);
        }
        std::stringstream buffer;
        buffer << file.rdbuf();
        std::string jsonContent = buffer.str();
        doc.Parse(jsonContent.c_str());
        if (doc.HasParseError()) {
            std::cerr << "Error parsing JSON: " << doc.GetParseError() << " at offset " << doc.GetErrorOffset() << std::endl;
        }
    }

    template<typename T>
    T get(const std::string& key) const {
        if (doc.HasMember(key.c_str())) {
            const Value& value = doc[key.c_str()];
            if constexpr (std::is_same<T, int>::value) {
                return value.GetInt();
            } else if constexpr (std::is_same<T, double>::value) {
                return value.GetDouble();
            } else if constexpr (std::is_same<T, std::string>::value) {
                return value.GetString();
            } else {
                throw std::runtime_error("Unsupported type for config value");
            }
        } else {
            throw std::runtime_error("Key not found in config: " + key);
        }
    }
};

class ConfigWriter {
private:
    StringBuffer buffer_;
    PrettyWriter<StringBuffer> writer_;
    bool finalized_ = false;

public:
    ConfigWriter() : writer_(buffer_) {
        writer_.StartObject();
    }

    ConfigWriter(const ConfigWriter&) = delete;
    ConfigWriter& operator=(const ConfigWriter&) = delete;

    template<typename T>
    ConfigWriter& set(const std::string& key, const T& value) {
        if (finalized_) {
            throw std::runtime_error("Cannot add values after saving the file.");
        }

        writer_.Key(key.c_str());

        if constexpr (std::is_same_v<T, bool>) {
            writer_.Bool(value);
        } else if constexpr (std::is_integral_v<T> && std::is_signed_v<T>) {
            writer_.Int64(static_cast<int64_t>(value));
        } else if constexpr (std::is_integral_v<T> && std::is_unsigned_v<T>) {
            writer_.Uint64(static_cast<uint64_t>(value));
        } else if constexpr (std::is_floating_point_v<T>) {
            writer_.Double(value);
        } else if constexpr (std::is_convertible_v<T, std::string>) {
            writer_.String(std::string(value).c_str());
        } else {
            static_assert(sizeof(T) == 0, "Unsupported type for ConfigWriter::set");
        }
        
        return *this;
    }

    void save(const std::string& filePath) {
        if (finalized_) {
            return; 
        }

        writer_.EndObject();
        finalized_ = true;

        std::ofstream file(filePath);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open file for writing: " + filePath);
        }

        file << buffer_.GetString();
    }
};