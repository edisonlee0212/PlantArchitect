/**
 * @brief Header for the main Catch2 source file.
 */

#include "catch_main.h"

#define CATCH_CONFIG_RUNNER
#include <catch2/catch.hpp>

int main(int argc, char *argv[])
{ const auto result{ Catch::Session().run(argc, argv) }; return result; }
