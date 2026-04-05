CC = gcc
MODE ?= release
OMP ?= 1
# Generic flags
LDFLAGS = -lm

CFLAGS_COMMON = -Wall -Wextra -std=c11 -MMD -MP
CFLAGS_DEBUG = -O0 -g
# Release: O3 para velocidad, ftree-vectorize para bucles, fno-fast-math para paridad numérica
CFLAGS_RELEASE = -O3 -DNDEBUG -fno-fast-math -ftree-vectorize
# Release fast: prioriza rendimiento (puede reducir paridad exacta)
CFLAGS_RELEASE_FAST = -O3 -DNDEBUG -ffast-math -fno-math-errno -fno-trapping-math -ftree-vectorize

ifeq ($(MODE),release)
  CFLAGS = $(CFLAGS_COMMON) $(CFLAGS_RELEASE)
else ifeq ($(MODE),release_fast)
  CFLAGS = $(CFLAGS_COMMON) $(CFLAGS_RELEASE_FAST)
else
  CFLAGS = $(CFLAGS_COMMON) $(CFLAGS_DEBUG)
endif

# OpenMP toggle (real): pass OMP=0 para build sin OpenMP
ifeq ($(OMP),1)
  CFLAGS += -fopenmp
  LDFLAGS += -fopenmp
endif

# Auto-detect Architecture
UNAME_M := $(shell uname -m)

ifeq ($(UNAME_M),aarch64)
  # Raspberry Pi 3B+/4 (64-bit)
  CFLAGS += -mcpu=cortex-a53 -mtune=cortex-a53 -DUSE_NEON
  # Note: 64-bit implies NEON support.
else ifeq ($(UNAME_M),armv7l)
  # Raspberry Pi 3 (32-bit Legacy) - Not recommended but supported
  CFLAGS += -mcpu=cortex-a53 -mfpu=neon-fp-armv8 -mfloat-abi=hard -DUSE_NEON
else
  # x86_64 Host (Dev/Test)
  CFLAGS += -march=native -mtune=native 
endif

# Manual Overrides (Optional)
ifeq ($(RPI_ARCH),rpi3)
  CFLAGS += -mcpu=cortex-a53 -mtune=cortex-a53 -DUSE_NEON
endif
ifeq ($(RPI_ARCH),rpi4)
  CFLAGS += -mcpu=cortex-a72 -mtune=cortex-a72 -DUSE_NEON
endif

TARGET_ENC = sorteny_compressor
TARGET_DEC = sorteny_decompressor
TARGET_TEST = sorteny_decoder_ops_test
SRC_DIR = src/c

COMMON_SRCS = \
  $(SRC_DIR)/sorteny_model.c \
  $(SRC_DIR)/sorteny_layers.c \
  $(SRC_DIR)/io_helpers.c

ENC_SRCS = $(SRC_DIR)/main.c $(COMMON_SRCS)
DEC_SRCS = $(SRC_DIR)/decompress.c $(COMMON_SRCS)
TEST_SRCS = $(SRC_DIR)/test_decoder_ops.c $(COMMON_SRCS)

ENC_OBJS = $(ENC_SRCS:.c=.o)
DEC_OBJS = $(DEC_SRCS:.c=.o)
TEST_OBJS = $(TEST_SRCS:.c=.o)
DEPS = $(ENC_OBJS:.o=.d) $(DEC_OBJS:.o=.d) $(TEST_OBJS:.o=.d)

.PHONY: all clean distclean run run_dec test_ops rpi3 rpi4 rpi3_fast rpi4_fast

# Permite usar '>' como prefijo de recetas en lugar de tabulador
.RECIPEPREFIX := >

all: $(TARGET_ENC) $(TARGET_DEC)

$(TARGET_ENC): $(ENC_OBJS)
> @echo Enlazando: $@
> $(CC) $(CFLAGS) -o $@ $(ENC_OBJS) $(LDFLAGS)

$(TARGET_DEC): $(DEC_OBJS)
> @echo Enlazando: $@
> $(CC) $(CFLAGS) -o $@ $(DEC_OBJS) $(LDFLAGS)

$(TARGET_TEST): $(TEST_OBJS)
> @echo Enlazando: $@
> $(CC) $(CFLAGS) -o $@ $(TEST_OBJS) $(LDFLAGS)

%.o: %.c
> @echo Compilando: $<
> $(CC) $(CFLAGS) -c $< -o $@

# Ruta de pesos: usar el set minimal por defecto
run: $(TARGET_ENC)
> ./$(TARGET_ENC) data/T31TCG_20230907T104629_5.8_512_512_2_1_0.raw 0.01 debug_dumps/Y_hat_c.bin weights/pesos_bin_minimal

run_dec: $(TARGET_DEC)
> ./$(TARGET_DEC) results/raspberry_results/output_c_20260105_165501.bin output/reconstructed_c.raw weights/pesos_ieec050_decoder 0.125

test_ops: $(TARGET_TEST)
> ./$(TARGET_TEST)

# --- SHORTCUTS PARA RASPBERRY ---

rpi3:
> $(MAKE) MODE=release RPI_ARCH=rpi3 OMP=0

rpi4:
> $(MAKE) MODE=release RPI_ARCH=rpi4 OMP=0

rpi3_fast:
> $(MAKE) MODE=release_fast RPI_ARCH=rpi3 OMP=0

rpi4_fast:
> $(MAKE) MODE=release_fast RPI_ARCH=rpi4 OMP=0

clean:
> @echo Limpiando...
> rm -f $(TARGET_ENC) $(TARGET_DEC) $(TARGET_TEST) $(ENC_OBJS) $(DEC_OBJS) $(TEST_OBJS) $(DEPS)

distclean: clean

-include $(DEPS)
