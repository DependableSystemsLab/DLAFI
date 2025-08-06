/* Library to parse command line arguments passed to the program for fault injection configuration.
 * */
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>

//#define dprintf(...) printf(__VA_ARGS__)
#define dprintf(...)

struct FI {
	int fi_tile_row;
	int fi_tile_col;
	int fi_pe_row;
	int fi_pe_col;
	int fi_type;
	int fi_bit_loc;
	int do_fi;
} FIParams;

bool isWS = false;

void resetFIParams() {
	FIParams.fi_tile_row = 0;
	FIParams.fi_tile_col = 0;
	FIParams.fi_pe_row = 0;
	FIParams.fi_pe_col = 0;
	FIParams.fi_type = 0;
	FIParams.fi_bit_loc = 0;
	FIParams.do_fi = 0;
}

uint8_t strContains(char* string, char* toFind)
{
    uint8_t slen = strlen(string);
    uint8_t tFlen = strlen(toFind);
    uint8_t found = 0;

    if( slen >= tFlen )
    {
        for(uint8_t s=0, t=0; s<slen; s++)
        {
            do{

                if( string[s] == toFind[t] )
                {
                    if( ++found == tFlen ) return s;
                    s++;
                    t++;
                }
                else { s -= found; found=0; t=0; }

              }while(found);
        }
        return 0;
    }
    else return 0;
}

// Function to parse arguments
void parseArguments(int argc, char** argv){

	resetFIParams();
	dprintf("Got %d args\n", argc);
	
	for (int i = 1; i < argc; i++) {

		dprintf("argv: %s\n", argv[i]);

		if (argv[i][0] == '-'){
			
			// It should be a boolean value.
			if (strContains(argv[i], "do-fi=") != 0) {
				int val = atoi(argv[i] + strContains(argv[i], "do-fi=") + 1);
				dprintf("do_fi found. Val= %d\n", val);
				FIParams.do_fi = (val > 0)?1:0;
			}
			if (strContains(argv[i], "fi-tile-row=") != 0) {
				int val = atoi(argv[i] + strContains(argv[i], "fi-tile-row=") + 1);
				dprintf("fi-tile-row found. Val= %d\n", val);
				FIParams.fi_tile_row = val;
			}
			if (strContains(argv[i], "fi-tile-col=") != 0) {
				int val = atoi(argv[i] + strContains(argv[i], "fi-tile-col=") + 1);
				dprintf("fi-tile-col found. Val= %d\n", val);
				FIParams.fi_tile_col = val;
			}
			if (strContains(argv[i], "fi-pe-row=") != 0) {
				int val = atoi(argv[i] + strContains(argv[i], "fi-pe-row=") + 1);
				dprintf("fi-pe-row found. Val= %d\n", val);
				FIParams.fi_pe_row = val;
			}
			if (strContains(argv[i], "fi-pe-col=") != 0) {
				int val = atoi(argv[i] + strContains(argv[i], "fi-pe-col=") + 1);
				dprintf("fi-pe-col found. Val= %d\n", val);
				FIParams.fi_pe_col = val;
			}
			if (strContains(argv[i], "fi-model=") != 0) {
				int val = atoi(argv[i] + strContains(argv[i], "fi-model=") + 1);
				dprintf("fault-model found. Val= %d\n", val);
				FIParams.fi_type = val;
			}
			if (strContains(argv[i], "fi-data=") != 0) {
				int val = atoi(argv[i] + strContains(argv[i], "fi-data=") + 1);
				dprintf("fault-data found. Val= %d\n", val);
				FIParams.fi_bit_loc = val;
			}
			if (strContains(argv[i], "WS") != 0) {
				dprintf("Found WS\n");
				isWS = true;
			}
		}
	}
}