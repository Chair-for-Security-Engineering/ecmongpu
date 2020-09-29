#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>
#include <netdb.h>
#include <sys/types.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <stdbool.h>
#include <unistd.h>

#include <gmp.h>

#define BUFSIZE 1000

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

int sock_connect(char *hostname, int port){
	int sockfd, numbytes;
	char buf[BUFSIZE];
	struct hostent *he;
	struct sockaddr_in srv_addr;
	if ((he=gethostbyname(hostname)) == NULL) {  /* get the host info */
		herror("gethostbyname");
		exit(1);
	}

	if ((sockfd = socket(AF_INET, SOCK_STREAM, 0)) == -1) {
		perror("socket");
		exit(1);
	}

	srv_addr.sin_family = AF_INET;      /* host byte order */
	srv_addr.sin_port = htons(port);    /* short, network byte order */
	srv_addr.sin_addr = *((struct in_addr *)he->h_addr);

	bzero(&(srv_addr.sin_zero), 8);     /* zero the rest of the struct */

	if (connect(sockfd, (struct sockaddr *)&srv_addr, sizeof(struct sockaddr)) == -1) {
		perror("connect");
		exit(1);
	}
	return sockfd;
}

int send_mpz(int sockfd, int id, mpz_t num){
	char sendbuf[BUFSIZE];
	int sendbytes = gmp_snprintf(sendbuf, BUFSIZE, " %d %Zi\n", id, num);
	return send(sockfd, sendbuf, sendbytes, 0);
}

int read_mpz(int sockfd, int *id, mpz_t f){
	int len = 0;
	char line[BUFSIZE];
	char *mpnum;
	if(readLine(sockfd, line, BUFSIZE) > 0){
		sscanf(line, "%d %m[0-9]", id, &mpnum);
		mpz_set_str(f, mpnum, 10);
		return 1;
	}
	return -1;
}

int main(int argc, char *argv[]){
	if (argc != 4) {
		fprintf(stderr,"usage: hostname port bitwidth\n");
		exit(1);
	}
	int bitwidth = atoi(argv[3]);

	gmp_randstate_t gmprand;
	gmp_randinit_default(gmprand);
	gmp_randseed_ui(gmprand, rand());

	/* Socket connection */
	int sockfd = sock_connect(argv[1], atoi(argv[2]));

	/* Generate 100 n=pq numbers of with width bitwidth and send over socket */
	mpz_t n, p, q;
	mpz_init(n);
	mpz_init(p);
	mpz_init(q);
	for(int i = 0; i < 100; i++ ){
		mpz_urandomb(p, gmprand, bitwidth/2);
		mpz_nextprime(p, p);
		mpz_urandomb(q, gmprand, bitwidth/2);
		mpz_nextprime(q, q);
		mpz_mul(n, p, q);

		if(send_mpz(sockfd, i, n) == -1){
			perror("send failed");
		}
	}

	/* Read results as gmp numbers and print */
	int id;
	mpz_t f;
	mpz_init(f);
	while(read_mpz(sockfd, &id, f) > 0){
		gmp_printf("%d %Zi\n", id, f);
	}
	close(sockfd);

	return 0;
}

