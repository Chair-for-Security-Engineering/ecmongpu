#include "ecc/naf.h"
#include "log.h"

#define TRPL_MASK 0x80
#define DBL_MASK 0x40
#define NEG_MASK 0x20
#define VALUE_MASK 0x1f

int to_naf(naf_t naf, size_t digits, mpz_t s, int w) {
	mpz_t mys;
	mpz_init(mys);
	mpz_set(mys, s);

	int twow = 1 << w;
	size_t i = 0;
	uint8_t naf_tmp[digits];

	/* Limb by limb version, avoids many shifts on large mys */
	/* Works on per_loop_shift bits at a time */
	size_t per_loop_shifts = GMP_LIMB_BITS - w;
	while (i < digits && mpz_cmp_ui(mys, 1) >= 0) {
		/* Work on least significant limb */
		mp_limb_t limb = mpz_getlimbn(mys, 0);

		/* Marks the last limb */
		bool last = (mpz_size(mys) == 1);

		signed long long int sub = 0;
		/* Tally for shifts in this limb */
		size_t shifts = 0;
		int8_t itmp;
		while (i < digits && shifts < per_loop_shifts) {
		  /* break if last limb and completely consumed */
		  if(limb == 0 && last) break;
			/* limb is odd */
			if (limb & 1) {
				itmp = limb % twow;
				if (itmp >= twow / 2) {
					itmp -= twow;
				}
				naf_tmp[i] = (abs(itmp) & VALUE_MASK) | DBL_MASK | (itmp < 0 ? NEG_MASK : 0x0);
				/* subtract all itmp's from this limb from mys later*/
				limb -= itmp;
				sub += (((signed long long int) itmp) << shifts);
			} else {
				naf_tmp[i] = 0 | DBL_MASK;
			}
			limb >>= 1;
			i++;
			shifts++;
		}
		/* GMP has no sub_si */
		if (sub > 0) {
		  mpz_sub_ui(mys, mys, sub);
    } else {
			mpz_add_ui(mys, mys, abs(sub));
    }
		mpz_tdiv_q_2exp(mys, mys, min(shifts, per_loop_shifts)); // mys >>= shifts;
	}

	// reverse array
  for (int c = i - 1, d = 0; c >= 0; c--, d++){
    naf[d] = naf_tmp[c];
  }

  // Append zero byte
	if (i < digits) {
	  naf[i] = 0x00;
	} else {
		LOG_FATAL("NAF transform failed");
		LOG_FATAL("Could not append zero byte delimiter at %d (%d digits)", i, digits);
		return -1;
	}

	if (mpz_cmp_ui(mys, 1) >= 0) {
		LOG_FATAL("NAF transform failed");
		LOG_FATAL("\ts: %Zi\n\ti: %d\n\tdigits: %d", mys, i, digits);
		return -1;
	} 
	return i+1;
}

void from_naf(mpz_t s, naf_t naf, size_t digits) {
	mpz_set_ui(s, 0);
	mpz_t tmp;
	mpz_init(tmp);
	int i = 0;
	while(naf[i] != 0x00) {
		if (naf[i] & DBL_MASK){
			mpz_add(s, s, s);
		}
		int8_t digit = (naf[i] & VALUE_MASK);
		if(naf[i] & NEG_MASK){
			mpz_sub_ui(s, s, digit);
		} else {
			mpz_add_ui(s, s, digit);
		}
		i++;
	}
}


__host__ __device__
void print_naf(naf_t a, size_t digits) {
	for (size_t i = 0; i < digits; i++) {
		printf("[%s%d]%s%s ", 
		    a[i] & NEG_MASK ? "-" : "",
		    a[i] & VALUE_MASK,
		    a[i] & DBL_MASK ? "D" : "",
		    a[i] & TRPL_MASK ? "T" : ""
		    );
	}
	printf("\n");
}
