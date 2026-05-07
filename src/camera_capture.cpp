#include "camera_capture.h"

#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <filesystem>
#include <sstream>
#include <string>
#include <vector>

namespace {

int RunCommand(const std::vector<std::string>& args, int* exit_code, std::string* error_message) {
  if (args.empty()) {
    if (error_message != nullptr) {
      *error_message = "Attempted to run an empty command.";
    }
    return -1;
  }

  std::vector<char*> argv;
  argv.reserve(args.size() + 1U);
  for (const std::string& arg : args) {
    argv.push_back(const_cast<char*>(arg.c_str()));
  }
  argv.push_back(nullptr);

  pid_t pid = fork();
  if (pid < 0) {
    if (error_message != nullptr) {
      *error_message = std::string("fork() failed: ") + std::strerror(errno);
    }
    return -1;
  }

  if (pid == 0) {
    execvp(argv[0], argv.data());
    _exit(errno == ENOENT ? 127 : 126);
  }

  int status = 0;
  if (waitpid(pid, &status, 0) < 0) {
    if (error_message != nullptr) {
      *error_message = std::string("waitpid() failed: ") + std::strerror(errno);
    }
    return -1;
  }

  if (WIFEXITED(status)) {
    *exit_code = WEXITSTATUS(status);
    return 0;
  }
  if (WIFSIGNALED(status)) {
    *exit_code = 128 + WTERMSIG(status);
    return 0;
  }

  *exit_code = status;
  return 0;
}

std::vector<std::string> MakeCaptureCommand(const std::string& executable, const CaptureOptions& options) {
  return {
      executable,
      "--nopreview",
      "--timeout",
      std::to_string(options.timeout_ms),
      "--width",
      std::to_string(options.width),
      "--height",
      std::to_string(options.height),
      "--encoding",
      "bmp",
      "--output",
      options.output_path,
  };
}

}  // namespace

bool CaptureStillBmp(const CaptureOptions& options, std::string* backend_used, std::string* error_message) {
  const std::vector<std::string> backends = {"rpicam-still", "libcamera-still"};

  for (const std::string& backend : backends) {
    int exit_code = -1;
    std::string run_error;
    const auto command = MakeCaptureCommand(backend, options);
    const int result = RunCommand(command, &exit_code, &run_error);

    if (result == 0 && exit_code == 0) {
      if (!std::filesystem::exists(options.output_path)) {
        if (error_message != nullptr) {
          *error_message = backend + " reported success but did not create " + options.output_path;
        }
        continue;
      }
      if (backend_used != nullptr) {
        *backend_used = backend;
      }
      if (error_message != nullptr) {
        error_message->clear();
      }
      return true;
    }

    if (result != 0) {
      if (error_message != nullptr) {
        *error_message = run_error;
      }
      continue;
    }

    if (exit_code == 127) {
      continue;
    }

    if (error_message != nullptr) {
      std::ostringstream oss;
      oss << backend << " exited with code " << exit_code;
      *error_message = oss.str();
    }
  }

  if (error_message != nullptr && error_message->empty()) {
    *error_message =
        "Could not find rpicam-still or libcamera-still on the Raspberry Pi. Install rpicam-apps or libcamera-apps.";
  }
  return false;
}
