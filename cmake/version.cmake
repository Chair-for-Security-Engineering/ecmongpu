# Use git revision in config
# Get the latest abbreviated commit hash of the working branch
execute_process(COMMAND git log --pretty=format:%h -n 1
                WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
                OUTPUT_VARIABLE GIT_COMMIT_HASH
                ERROR_QUIET
	              OUTPUT_STRIP_TRAILING_WHITESPACE)

if("${GIT_COMMIT_HASH}" STREQUAL "")
	set(GIT_BRANCH "release")
  set(GIT_DIRTY "")
  set(GIT_COMMIT_HASH ${VERSION})
else()
  # Mark dirty working directory
  execute_process(
          COMMAND bash -c "git diff --quiet --exit-code || echo +"
          WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
          OUTPUT_VARIABLE GIT_DIFF
	        OUTPUT_STRIP_TRAILING_WHITESPACE)
  # Get the current working branch
  execute_process(
          COMMAND git rev-parse --abbrev-ref HEAD
          WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
          OUTPUT_VARIABLE GIT_BRANCH
	        OUTPUT_STRIP_TRAILING_WHITESPACE)

endif()

#Generate version.h
configure_file(
        include/version.h.in
        ${CMAKE_CURRENT_BINARY_DIR}/generated/version.h
)
