# SPDX-License-Identifier: 0BSD

{
  description = "pyAmpliCol contributor development environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    { nixpkgs, rust-overlay, ... }:
    let
      systems = [
        "aarch64-darwin"
        "x86_64-darwin"
        "aarch64-linux"
        "x86_64-linux"
      ];
      forAllSystems = nixpkgs.lib.genAttrs systems;
    in
    {
      devShells = forAllSystems (
        system:
        let
          pkgs = import nixpkgs {
            inherit system;
            overlays = [ (import rust-overlay) ];
          };
          inherit (pkgs) lib stdenv;

          rustToolchain = pkgs.rust-bin.fromRustupToolchainFile ./rust-toolchain.toml;

          # Project, test, and candidate Python packages are deliberately absent:
          # dependencies/install_dependencies.py owns that lock-controlled venv.
          python = pkgs.python311.withPackages (
            pythonPackages: with pythonPackages; [
              pip
              setuptools
              wheel
            ]
          );

          nativeLibraries = with pkgs; [
            gmp
            libffi
            libmpc
            mpfr
            openssl
            stdenv.cc.cc.lib
            zlib
          ];

          platformTools =
            lib.optionals stdenv.isLinux [
              pkgs.binutils
              pkgs.patchelf
            ]
            ++ lib.optionals stdenv.isDarwin [
              pkgs.darwin.cctools
              pkgs.libiconv
            ];
        in
        {
          default = pkgs.mkShell {
            packages =
              (with pkgs; [
                bash
                cacert
                cmake
                coreutils
                curl
                diffutils
                file
                findutils
                gh
                ghostscript
                git
                gnumake
                gnugrep
                gnused
                gnutar
                gfortran
                gzip
                just
                jq
                lhapdf
                ninja
                openssh
                patch
                pkg-config
                poppler-utils
                python
                ripgrep
                rustToolchain
                stdenv.cc
                texliveFull
                unzip
                which
                xz
                zip
                zstd
              ])
              ++ platformTools;

            buildInputs = nativeLibraries;

            PYAMPLICOL_DEV_PYTHON = ".venv/bin/python";
            PIP_DISABLE_PIP_VERSION_CHECK = "1";
            PYTHONNOUSERSITE = "1";

            # These paths support native wheels and generated SDK consumers in
            # a Nix shell without changing release-artifact link metadata.
            LD_LIBRARY_PATH = lib.optionalString stdenv.isLinux (
              lib.makeLibraryPath nativeLibraries
            );
            NIX_LD_LIBRARY_PATH = lib.optionalString stdenv.isLinux (
              lib.makeLibraryPath nativeLibraries
            );

            shellHook = ''
              echo "pyAmpliCol developer shell (Python 3.11, Rust 1.89, native SDK, TeX)"
              echo "Run 'just dev-install' to create the pinned contributor environment."
            '';
          };
        }
      );
    };
}
