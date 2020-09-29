#ifndef INPUT_INPUT_H
#define INPUT_INPUT_H

#include "config/config.h"

#ifdef __cplusplus
extern "C" {
#endif
/**
 * Parses an input line and executes the necessary functions
 * @param line		Line to parse
 * @param len		Length of the line
 * @param config	Runtime configuration
 * @return
 */
int parse_input(char *line, size_t len, run_config config);

/**
 * Starts the TCP-listener thread according to the configuration.
 *
 */
void *server_thread_run(void *configv);

/**
 * Starts the file-based input thread according to the configuration.
 */
void *file_input_thread_run(void *configv);


#ifdef __cplusplus
}
#endif

#endif /* INPUT_INPUT_H */
