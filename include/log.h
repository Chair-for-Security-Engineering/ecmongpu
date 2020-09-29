/*
	co-ecm
	Copyright (C) 2018  Jonas Wloka

	This program is free software: you can redistribute it and/or modify
	it under the terms of the GNU General Public License as published by
			the Free Software Foundation, either version 3 of the License, or
	(at your option) any later version.

	This program is distributed in the hope that it will be useful,
			but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	GNU General Public License for more details.

	You should have received a copy of the GNU General Public License
	along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef CO_ECM_LOG_H
#define CO_ECM_LOG_H

#include <pthread.h>
#include "build_config.h"
#include "string.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Log functions with level.
 *
 * @param file 		Source filename.
 * @param line 		Source file line number.
 * @param format 	printf-like format string.
 * @param ...		Arguments to printf-format string.
 */

void _log_verbose(const char *const file, const int line, const char *const format, ...);

void _log_debug(const char *const file, const int line, const char *const format, ...);

void _log_info(const char *const file, const int line, const char *const format, ...);

void _log_warning(const char *const file, const int line, const char *const format, ...);

void _log_error(const char *const file, const int line, const char *const format, ...);

void _log_fatal(const char *const file, const int line, const char *const format, ...);

/**
 * Opens a global file as logfile.
 *
 * @param log_path 	Filename of the log file.
 * @return
 */
int logfile_open(const char *const log_path);

void log_set_color(bool use_color);

/**
 * Closes the global logfile.
 */
void logfile_close();

/**
 * Set loglevel at runtime.
 *
 * @param level
 */
void loglevel_set(uint8_t level);

/**
 * Get the string representation of the supplied level.
 *
 * @param level
 * @return
 */
const char *log_level_name(const int level);

#define LOG_LEVEL_VERBOSE 1
#define LOG_LEVEL_DEBUG   2
#define LOG_LEVEL_INFO    3
#define LOG_LEVEL_WARNING 4
#define LOG_LEVEL_ERROR   5
#define LOG_LEVEL_FATAL   6
#define LOG_LEVEL_NONE    0xFF


#define __FILENAME__ (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)

/**
 * Define level specific macros LOG_... and LOG_LEVEL_..._ENABLED to be used as #ifdef guards.
 */

#if (LOG_LEVEL <= LOG_LEVEL_VERBOSE)
#define LOG_VERBOSE(...) _log_verbose(__FILENAME__, __LINE__, __VA_ARGS__);
#define LOG_LEVEL_VERBOSE_ENABLED 1
#else
#define LOG_VERBOSE(...)
#endif

#if (LOG_LEVEL <= LOG_LEVEL_DEBUG)
#define LOG_DEBUG(...) _log_debug(__FILENAME__, __LINE__, __VA_ARGS__);
#define LOG_LEVEL_DEBUG_ENABLED 1
#else
#define LOG_DEBUG(...)
#endif

#if (LOG_LEVEL <= LOG_LEVEL_INFO)
#define LOG_INFO(...) _log_info(__FILENAME__, __LINE__, __VA_ARGS__);
#define LOG_LEVEL_INFO_ENABLED 1
#else
#define LOG_INFO(...)
#endif

#if (LOG_LEVEL <= LOG_LEVEL_WARNING)
#define LOG_WARNING(...) _log_warning(__FILENAME__, __LINE__, __VA_ARGS__);
#define LOG_LEVEL_WARNING_ENABLED 1
#else
#define LOG_WARNING(...)
#endif

#if (LOG_LEVEL <= LOG_LEVEL_ERROR)
#define LOG_ERROR(...) _log_error(__FILENAME__, __LINE__, __VA_ARGS__);
#define LOG_LEVEL_ERROR_ENABLED 1
#else
#define LOG_ERROR(...)
#endif

#if (LOG_LEVEL <= LOG_LEVEL_FATAL)
#define LOG_FATAL(...) _log_fatal(__FILENAME__, __LINE__, __VA_ARGS__);
#define LOG_LEVEL_FATAL_ENABLED 1
#else
#define LOG_FATAL(...)
#endif

#ifdef __cplusplus
};
#endif


#endif //CO_ECM_LOG_H
