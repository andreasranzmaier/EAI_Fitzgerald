FROM mcr.microsoft.com/devcontainers/cpp:ubuntu-22.04

# This container cross-compiles the application for Raspberry Pi/aarch64.
# RTIMULib is therefore staged into the aarch64 sysroot so the linker sees
# the same target-side library that is installed on the Pi.
RUN set -eux; \
    . /etc/os-release; \
    native_arch="$(dpkg --print-architecture)"; \
    if [ "$native_arch" != "arm64" ]; then \
      dpkg --add-architecture arm64; \
      for source_list in /etc/apt/sources.list /etc/apt/sources.list.d/*.list; do \
        [ -f "$source_list" ] || continue; \
        sed -i -E \
          -e "/^deb \[/ { /arch=/! s#^deb \[#deb [arch=${native_arch} #; }" \
          -e "/^deb [^[]/ s#^deb #deb [arch=${native_arch}] #" \
          "$source_list"; \
      done; \
      if ls /etc/apt/sources.list.d/*.sources >/dev/null 2>&1; then \
        for source_file in /etc/apt/sources.list.d/*.sources; do \
          awk -v native_arch="$native_arch" 'BEGIN { in_stanza = 0; saw_arch = 0 } \
               /^Types:/ { in_stanza = 1; saw_arch = 0 } \
               /^Architectures:/ { print "Architectures: " native_arch; saw_arch = 1; next } \
               /^$/ { if (in_stanza && !saw_arch) print "Architectures: " native_arch; print; in_stanza = 0; saw_arch = 0; next } \
               { print } \
               END { if (in_stanza && !saw_arch) print "Architectures: " native_arch }' "$source_file" > "$source_file.tmp"; \
          mv "$source_file.tmp" "$source_file"; \
        done; \
      fi; \
      printf '%s\n' \
        'Types: deb' \
        'URIs: http://ports.ubuntu.com/ubuntu-ports' \
        "Suites: ${VERSION_CODENAME} ${VERSION_CODENAME}-updates ${VERSION_CODENAME}-backports ${VERSION_CODENAME}-security" \
        'Components: main restricted universe multiverse' \
        'Architectures: arm64' \
        'Signed-By: /usr/share/keyrings/ubuntu-archive-keyring.gpg' \
        > /etc/apt/sources.list.d/ubuntu-arm64.sources; \
    fi

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake ninja-build \
    rsync openssh-client sshpass tmux \
    gdb-multiarch \
    gcc-aarch64-linux-gnu g++-aarch64-linux-gnu \
    git curl wget unzip ca-certificates \
    python3 python3-pip python3-venv python3-dev \
    libopencv-dev \
 && mkdir -p /tmp/rtimulib-arm64-debs /tmp/rtimulib-arm64-root \
 && cd /tmp/rtimulib-arm64-debs \
 && apt-get download librtimulib7:arm64 librtimulib-dev:arm64 \
 && for deb in *.deb; do dpkg-deb -x "$deb" /tmp/rtimulib-arm64-root; done \
 && mkdir -p /usr/aarch64-linux-gnu/include /usr/aarch64-linux-gnu/lib \
 && cp -a /tmp/rtimulib-arm64-root/usr/include/RT*.h /tmp/rtimulib-arm64-root/usr/include/IMUDrivers /usr/aarch64-linux-gnu/include/ \
 && cp -a /tmp/rtimulib-arm64-root/usr/lib/libRTIMULib.so* /usr/aarch64-linux-gnu/lib/ \
 && printf '%s\n' '#include <RTIMULib.h>' 'int main() { return 0; }' > /tmp/rtimulib-link-check.cpp \
 && aarch64-linux-gnu-g++ \
      -I/usr/aarch64-linux-gnu/include \
      /tmp/rtimulib-link-check.cpp \
      -L/usr/aarch64-linux-gnu/lib \
      -Wl,--no-as-needed -lRTIMULib \
      -o /tmp/rtimulib-link-check \
 && rm -rf \
      /tmp/rtimulib-arm64-debs \
      /tmp/rtimulib-arm64-root \
      /tmp/rtimulib-link-check \
      /tmp/rtimulib-link-check.cpp \
      /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

WORKDIR /workspace
