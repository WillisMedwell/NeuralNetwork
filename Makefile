SRC := src/Main.cpp
INC := -I include/

LOCAL_DEBUG_BUILD := clang++ -std=c++2b -O3 -mtune=native -fexperimental-library -g

ENABLED_WARNINGS := -Wall -Wextra -Wpedantic -Wcast-align -Wcast-qual -Wctor-dtor-privacy -Wformat=2 -Winit-self -Wmissing-declarations -Wmissing-include-dirs -Woverloaded-virtual -Wredundant-decls -Wshadow -Wsign-conversion -Wsign-promo -Wstrict-overflow=5 -Wswitch-default -Wundef -Wno-unused -Wpedantic -Wconversion

all: 
	$(LOCAL_DEBUG_BUILD) $(ENABLED_WARNINGS) $(DISABLED_WARNINGS) $(INC) -o build/app.exe $(SRC)
	./build/app.exe

run:
	./build/app.exe
