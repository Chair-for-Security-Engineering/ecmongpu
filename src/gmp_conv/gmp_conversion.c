#include "mp/gmp_conversion.h"
#include "log.h"

#ifdef __cplusplus
extern "C" {
#endif

#define MP_LIMBS_PER_GMP ((GMP_LIMB_BITS + LIMB_BITS - 1)/LIMB_BITS)

void mpz_to_mp(mp_t a, const mpz_t b) {
	mpz_to_mp_limbs(a, b, LIMBS);
}

void mpz_to_mp_limbs(mp_t a, const mpz_t b, const size_t limbs) {
	mp_set_ui(a, 0);
	for (size_t i = 0; i < limbs; i += MP_LIMBS_PER_GMP) {
		mp_limb_t tmp;
		tmp = mpz_getlimbn(b, i / MP_LIMBS_PER_GMP);
#if (GMP_LIMB_BITS / LIMB_BITS) > 1
		for (int j = 0; j < MP_LIMBS_PER_GMP; j++) {
			if (i + j > limbs - 1 && tmp != 0) {
				LOG_ERROR("GMP to MP Conversion error: Insufficient limbs");
			}
			if (i + j < limbs) a[i + j] = tmp & LIMB_MASK;
			tmp >>= LIMB_BITS;
		}
#else
		a[i] = tmp & LIMB_MASK;
#endif
	}
}

void mp_to_mpz(mpz_t a, const mp_t b) {
	mpz_set_ui(a, 0);
	for (int i = LIMBS - 1; i >= 0; i--) {
		mpz_mul_2exp(a, a, LIMB_BITS);
		mpz_add_ui(a, a, b[i]);
	}
}

void mp_printf(const char *format, const mp_t a) {
	mpz_t tmp;
	mpz_init(tmp);
	mp_to_mpz(tmp, a);
	gmp_printf(format, tmp);
	mpz_clear(tmp);
}


#ifdef __cplusplus
}
#endif    
