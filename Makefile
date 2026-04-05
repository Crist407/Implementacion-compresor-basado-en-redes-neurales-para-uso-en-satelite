CC = gcc
MODE ?= release
# Generic flags
LDFLAGS = -lm -fopenmp

CFLAGS_COMMON = -Wall -Wextra -std=c11 -MMD -MP -fopenmp
CFLAGS_DEBUG = -O0 -g
# Release: O3 para velocidad, ftree-vectorize para bucles, fno-fast-math para paridad numérica
CFLAGS_RELEASE = -O3 -DNDEBUG -fno-fast-math -ftree-vectorize

ifeq ($(MODE),release)
  CFLAGS = $(CFLAGS_COMMON) $(CFLAGS_RELEASE)
else
  CFLAGS = $(CFLAGS_COMMON) $(CFLAGS_DEBUG)
endif

# OpenMP toggle: pass OMP=1 to enable
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

# --- Directorios y fuentes comunes ---
SRC_DIR = src/c
COMMON_SRCS = \
  $(SRC_DIR)/sorteny_model.c \
  $(SRC_DIR)/sorteny_layers.c \
  $(SRC_DIR)/io_helpers.c

# --- Compresor ---
TARGET = sorteny_compressor
SRCS = $(SRC_DIR)/main.c $(COMMON_SRCS)
OBJS = $(SRCS:.c=.o)
DEPS = $(OBJS:.o=.d)

# --- Descompresor ---
TARGET_DECOMP = sorteny_decompressor
SRCS_DECOMP = $(SRC_DIR)/main_decompress.c $(COMMON_SRCS)
OBJS_DECOMP = $(SRCS_DECOMP:.c=.o)
DEPS_DECOMP = $(OBJS_DECOMP:.o=.d)

.PHONY: all clean distclean run rpi3 rpi4 decompress

# Permite usar '>' como prefijo de recetas en lugar de tabulador
.RECIPEPREFIX := >

all: $(TARGET) $(TARGET_DECOMP)

$(TARGET): $(OBJS)
> @echo Enlazando: $@
> $(CC) $(CFLAGS) -o $@ $(OBJS) $(LDFLAGS)

decompress: $(TARGET_DECOMP)

$(TARGET_DECOMP): $(OBJS_DECOMP)
> @echo Enlazando: $@
> $(CC) $(CFLAGS) -o $@ $(OBJS_DECOMP) $(LDFLAGS)

%.o: %.c
> @echo Compilando: $<
> $(CC) $(CFLAGS) -c $< -o $@

run_dec: $(TARGET_DECOMP)
> ./$(TARGET_DECOMP) output/latent.bin output/reconstructed_c.raw weights/decoder 0.125

# Default run shortcut
run: $(TARGET)
> ./$(TARGET) data/T31TCG_20230907T104629_5.8_512_512_2_1_0.raw 0.1 output/latent.bin weights/encoder 0.125

# --- SHORTCUTS PARA RASPBERRY ---

rpi3:
> $(MAKE) MODE=release RPI_ARCH=rpi3 OMP=0

rpi4:
> $(MAKE) MODE=release RPI_ARCH=rpi4 OMP=0

clean:
> @echo Limpiando...
> rm -f $(TARGET) $(TARGET_DECOMP) $(OBJS) $(OBJS_DECOMP) $(DEPS) $(DEPS_DECOMP)

distclean: clean

-include $(DEPS) $(DEPS_DECOMP)