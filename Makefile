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

TARGET = sorteny_compressor
SRC_DIR = src/c
SRCS = \
  $(SRC_DIR)/main.c \
  $(SRC_DIR)/sorteny_model.c \
  $(SRC_DIR)/sorteny_layers.c \
  $(SRC_DIR)/io_helpers.c
OBJS = $(SRCS:.c=.o)
DEPS = $(OBJS:.o=.d)

.PHONY: all clean distclean run rpi3 rpi4

# Permite usar '>' como prefijo de recetas en lugar de tabulador
.RECIPEPREFIX := >

all: $(TARGET)

$(TARGET): $(OBJS)
> @echo Enlazando: $@
> $(CC) $(CFLAGS) -o $@ $(OBJS) $(LDFLAGS)

%.o: %.c
> @echo Compilando: $<
> $(CC) $(CFLAGS) -c $< -o $@

# Ruta de pesos: usar el set minimal por defecto
run: $(TARGET)
> ./$(TARGET) data/T31TCG_20230907T104629_5.8_512_512_2_1_0.raw 0.01 debug_dumps/Y_hat_c.bin weights/pesos_bin_minimal

# --- SHORTCUTS PARA RASPBERRY ---

rpi3:
> $(MAKE) MODE=release RPI_ARCH=rpi3 OMP=0

rpi4:
> $(MAKE) MODE=release RPI_ARCH=rpi4 OMP=0

clean:
> @echo Limpiando...
> rm -f $(TARGET) $(OBJS) $(DEPS)

distclean: clean

-include $(DEPS)