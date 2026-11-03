# Writes cps_version.hpp from the state of the tree at build time. Run by the
# always-run cps_version target in CMakeLists.txt, so a commit or an edit shows
# up without reconfiguring. The file is only written when its contents change:
# its timestamp is what the compiler dependencies see, so an unchanged tree
# rebuilds nothing.
#
# Expects PROJECT_VERSION, SOURCE_DIR and OUTPUT, and GIT_EXECUTABLE if git was
# found.

set(hash "unknown")
set(dirty 0)
if(GIT_EXECUTABLE)
  execute_process(
    COMMAND ${GIT_EXECUTABLE} rev-parse --short=12 HEAD
    WORKING_DIRECTORY ${SOURCE_DIR}
    OUTPUT_VARIABLE head
    RESULT_VARIABLE head_status
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET)
  if(head_status EQUAL 0 AND NOT head STREQUAL "")
    # Dirty means the code changed, not that a benchmark ran. Results and the
    # reports built from them are tracked, so without these exclusions a run
    # would mark the tree dirty and block the next one. Untracked files do not
    # count: a new source file only takes part once a tracked CMakeLists.txt
    # names it.
    execute_process(
      COMMAND ${GIT_EXECUTABLE} diff --quiet HEAD -- .
              ":(exclude)experiments/results"
              ":(exclude)report/figures"
              ":(exclude)report/tables"
              ":(exclude)report/numbers.tex"
              ":(exclude)docs/generated_numbers.md"
              ":(exclude)report/main.pdf"
              ":(exclude)report_debug/debug_report.pdf"
      WORKING_DIRECTORY ${SOURCE_DIR}
      RESULT_VARIABLE diff_status
      OUTPUT_QUIET
      ERROR_QUIET)
    # 0 is clean and 1 is different. Anything else means git could not tell,
    # which is the same as not knowing the tree.
    if(diff_status EQUAL 0)
      set(hash "${head}")
    elseif(diff_status EQUAL 1)
      set(hash "${head}")
      set(dirty 1)
    endif()
  endif()
endif()

set(stamp "${hash}")
if(dirty)
  set(stamp "${hash}-dirty")
endif()

set(contents "// Generated at build time by cmake/write_version_header.cmake. Do not edit.
#pragma once

#define CPS_VERSION \"${PROJECT_VERSION}\"
#define CPS_GIT_HASH \"${hash}\"
#define CPS_GIT_DIRTY ${dirty}
// The commit value recorded on a result row.
#define CPS_GIT_STAMP \"${stamp}\"
// What --version prints.
#define CPS_VERSION_STRING \"${PROJECT_VERSION} (${stamp})\"
")

set(current "")
if(EXISTS "${OUTPUT}")
  file(READ "${OUTPUT}" current)
endif()
if(NOT current STREQUAL contents)
  file(WRITE "${OUTPUT}" "${contents}")
endif()
