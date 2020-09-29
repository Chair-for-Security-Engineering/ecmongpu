#include <netinet/in.h>
#include <sys/socket.h>
#include <errno.h>
#include <unistd.h>
#include <stdlib.h>

#include "input/input.h"
#include "log.h"


static ssize_t readLine(int fd, char *buffer, size_t n) {
	ssize_t numRead;                    /* # of bytes fetched by last read() */
	size_t totRead;                     /* Total bytes read so far */
	char *buf;
	char ch;

	if (n <= 0 || buffer == NULL) {
		errno = EINVAL;
		return -1;
	}

	buf = buffer;                       /* No pointer arithmetic on "void *" */

	totRead = 0;
	while (true) {
		numRead = read(fd, &ch, 1);

		if (numRead == -1) {
			if (errno == EINTR)         /* Interrupted --> restart read() */
				continue;
			else
				return -1;              /* Some other error */

		} else if (numRead == 0) {      /* EOF */
			if (totRead == 0)           /* No bytes read; return 0 */
				return 0;
			else                        /* Some bytes read; add '\0' */
				break;

		} else {                        /* 'numRead' must be 1 if we get here */
			if (totRead < n - 1) {      /* Discard > (n - 1) bytes */
				totRead++;
				*buf++ = ch;
			}

			if (ch == '\n')
				break;
		}
	}

	*buf = '\0';
	return totRead;
}

void *server_thread_run(void *configv) {
	run_config config = (run_config) configv;

	if (config->server.port == 0) {
		LOG_FATAL("No server Port set.");
		exit(EXIT_FAILURE);
	}

	struct sockaddr_in address;
	int opt = 1;
	int addrlen = sizeof(address);

	// Creating socket file descriptor
	if ((config->server.socket = socket(AF_INET, SOCK_STREAM, 0)) == 0) {
		perror("socket failed");
		exit(EXIT_FAILURE);
	}

	// Forcefully attaching socket to the port 8080
	if (setsockopt(config->server.socket, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &opt, sizeof(opt))) {
		perror("setsockopt");
		exit(EXIT_FAILURE);
	}
	address.sin_family = AF_INET;
	address.sin_addr.s_addr = INADDR_ANY;
	address.sin_port = htons(config->server.port);

	// Forcefully attaching socket to the port 8080
	if (bind(config->server.socket, (struct sockaddr *) &address, sizeof(address)) < 0) {
		perror("bind failed");
		exit(EXIT_FAILURE);
	}

	if (listen(config->server.socket, 1) < 0) {
		perror("listen");
		exit(EXIT_FAILURE);
	}

	while (config->run) {
		LOG_INFO("Waiting for client connection...")
		if ((config->server.client_socket = accept(config->server.socket,
												   (struct sockaddr *) &address,
												   (socklen_t *) &addrlen)) < 0) {
			perror("accept");
			exit(EXIT_FAILURE);
		}
		LOG_INFO("New client connected")

		char line[1024] = {0};
		ssize_t len;
		while ((len = readLine(config->server.client_socket, line, 1024)) > 0) {
			if (!parse_input(line, len, config)) {
				LOG_WARNING("Malformed input: %s", line);
			}
		}
	}

	return NULL;

}
