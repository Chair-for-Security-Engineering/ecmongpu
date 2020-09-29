#include <stdio.h>
#include <unistd.h>
#include <stdarg.h>
#include <time.h>
#include <stdbool.h>
#include <sys/time.h>
#include "log.h"
#include "gmp.h"


#define RED   "\x1B[31m"
#define GRN   "\x1B[32m"
#define YEL   "\x1B[33m"
#define BLU   "\x1B[34m"
#define MAG   "\x1B[35m"
#define CYN   "\x1B[36m"
#define WHT   "\x1B[37m"
#define RESET "\x1B[0m"

pthread_mutex_t mutex_log = PTHREAD_MUTEX_INITIALIZER;

FILE *g_log_file;
bool g_log_use_color = false;

uint8_t g_log_level = 0;

char _logline_buf[4096];

void logfile_close(void) {
	if (g_log_file) {
		fclose(g_log_file);
	}
}


static const char *log_level_names[] = {
		"INVALID",
		"VERBOSE",
		"DEBUG",
		"INFO",
		"WARNING",
		"ERROR",
		"FATAL",
		"NONE"
};

static void printc(const char * col){
	if(g_log_use_color) printf("%s", col);
}

const char *log_level_name(const int level) {
	if (level < 0) return "INVALID";
	return log_level_names[(level > 7 ? 7 : level)];
}

static void _log_write(const char *line) {
	printf("%s", line);
	if (g_log_file) {
		fprintf(g_log_file, "%s", line);
	}
}


static void _log_time(void) {
	time_t timer;
	struct tm *tm_info;

	time(&timer);
	tm_info = localtime(&timer);

	struct timeval tval;
	gettimeofday(&tval, NULL);

	strftime(_logline_buf, 4069, "%Y-%m-%d %H:%M:%S.", tm_info);
	_log_write(_logline_buf);
	snprintf(_logline_buf, 4096, "%06ld", tval.tv_usec);
	_log_write(_logline_buf);
}

void
_log_common(const char *const level, const char *const file, const int line, const char *const format, va_list args) {
	_log_time();
#ifdef NDEBUG
	(void)file;
	(void)line;
	  snprintf(_logline_buf, 4096, " [%s]\t", level);
#else
	snprintf(_logline_buf, 4096, " @%s:%d [%s]\t", file, line, level);
#endif
	_log_write(_logline_buf);
	gmp_vsnprintf(_logline_buf, 4096, format, args);
	_log_write(_logline_buf);
	_log_write("\n");
}

void _log_verbose(const char *const file, const int line, const char *const format, ...) {
	if (!(g_log_level > LOG_LEVEL_VERBOSE)) {
    pthread_mutex_lock(&mutex_log);
	  printc(YEL);
	  va_list args;
	  va_start(args, format);
	  _log_common("VERBOSE", file, line, format, args);
	  va_end(args);
	  printc(RESET);
    pthread_mutex_unlock(&mutex_log);
  }
}

void _log_debug(const char *const file, const int line, const char *const format, ...) {
	if (!(g_log_level > LOG_LEVEL_DEBUG)) {
    pthread_mutex_lock(&mutex_log);
	  printc(YEL);
	  va_list args;
	  va_start(args, format);
	  _log_common("DEBUG", file, line, format, args);
	  va_end(args);
	  printc(RESET);
    pthread_mutex_unlock(&mutex_log);
  }
}

void _log_info(const char *const file, const int line, const char *const format, ...) {
	if (!(g_log_level > LOG_LEVEL_INFO)) {
    pthread_mutex_lock(&mutex_log);
	  printc(GRN);
	  va_list args;
	  va_start(args, format);
	  _log_common("INFO", file, line, format, args);
	  va_end(args);
	  printc(RESET);
    pthread_mutex_unlock(&mutex_log);
  }
}

void _log_warning(const char *const file, const int line, const char *const format, ...) {
	if (!(g_log_level > LOG_LEVEL_WARNING)) {
    pthread_mutex_lock(&mutex_log);
	  printc(MAG);
	  va_list args;
	  va_start(args, format);
	  _log_common("WARN", file, line, format, args);
	  va_end(args);
	  printc(RESET);
    pthread_mutex_unlock(&mutex_log);
  }
}

void _log_error(const char *const file, const int line, const char *const format, ...) {
	if (!(g_log_level > LOG_LEVEL_ERROR)) {
    pthread_mutex_lock(&mutex_log);
	  printc(RED);
	  va_list args;
	  va_start(args, format);
	  _log_common("ERROR", file, line, format, args);
	  va_end(args);
	  printc(RESET);
    pthread_mutex_unlock(&mutex_log);
  }
}

void _log_fatal(const char *const file, const int line, const char *const format, ...) {
	if (!(g_log_level > LOG_LEVEL_FATAL)) {
    pthread_mutex_lock(&mutex_log);
	  printc(RED);
	  va_list args;
	  va_start(args, format);
	  _log_common("FATAL", file, line, format, args);
	  va_end(args);
	  printc(RESET);
    pthread_mutex_unlock(&mutex_log);
  }
}

int logfile_open(const char *const log_path) {
	g_log_file = fopen(log_path, "a");
	if (g_log_file == NULL) {
		LOG_WARNING("Failed to open log file %s", log_path);
		return -1;
	}
	return 0;
}

void log_set_color(bool use_color) {
	g_log_use_color = use_color;
}

void loglevel_set(uint8_t level) {
	g_log_level = level;
	if (level > LOG_LEVEL) {
		LOG_WARNING("Runtime log level less verbose than compile time setting.");
		LOG_WARNING("Expect performance impact.");
	}
	if (level < LOG_LEVEL) {
		LOG_WARNING("Runtime log level cannot be set lower than compile time setting.");
		LOG_WARNING("Continuing with level %i", LOG_LEVEL);
	}
}
