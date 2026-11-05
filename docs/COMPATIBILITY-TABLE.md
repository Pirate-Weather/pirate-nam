# NOAA NAM awphys records in WOOF-NAM files

One row per record of NOAA's nam.tHHz.awphys06.tm00.grib2 (454 records), checked against a WOOF-NAM f06 of the same case (2025-03-14 12Z) with `tools/grib2_compat_table.py --plain`. "identical keys" means every GRIB2 key a decoder selects on (name, level, step, units, codes, grid and NOAA's precision) equals NOAA's; only the generating process number (254, WOOF-NAM, where NOAA's is 84) differs by design. See COMPATIBILITY.md.

Summary: 444 present with identical keys, 0 present with differing keys, 10 not in the checked file.

| rec | NAM record (name, level) | present | identical keys | notes |
|---|---|---|---|---|
| 1 | PRMSL mean sea level | yes | yes |  |
| 2 | REFD 1 hybrid level | yes | yes |  |
| 3 | REFC entire atmosphere (considered as a single layer) | yes | yes |  |
| 4 | VIS surface | yes | yes |  |
| 5 | VIS cloud top | no | - | not produced (see COMPATIBILITY.md, 'Not in the files') |
| 6 | BMIXL 1 hybrid level | no | - | not produced (see COMPATIBILITY.md, 'Not in the files') |
| 7 | HGT planetary boundary layer | yes | yes |  |
| 8.1 | UGRD planetary boundary layer | yes | yes |  |
| 8.2 | VGRD planetary boundary layer | yes | yes |  |
| 9 | VRATE planetary boundary layer | yes | yes |  |
| 10 | GUST surface | yes | yes |  |
| 11 | RETOP entire atmosphere (considered as a single layer) | yes | yes |  |
| 12 | DZDT 5 mb | yes | yes |  |
| 13 | DZDT 10 mb | yes | yes |  |
| 14 | DZDT 20 mb | yes | yes |  |
| 15 | HGT 50 mb | yes | yes |  |
| 16 | TMP 50 mb | yes | yes |  |
| 17 | RH 50 mb | yes | yes |  |
| 18 | VVEL 50 mb | yes | yes |  |
| 19 | DZDT 50 mb | yes | yes |  |
| 20.1 | UGRD 50 mb | yes | yes |  |
| 20.2 | VGRD 50 mb | yes | yes |  |
| 21 | TKE 50 mb | yes | yes |  |
| 22 | HGT 75 mb | yes | yes |  |
| 23 | TMP 75 mb | yes | yes |  |
| 24 | RH 75 mb | yes | yes |  |
| 25 | VVEL 75 mb | yes | yes |  |
| 26 | DZDT 75 mb | yes | yes |  |
| 27.1 | UGRD 75 mb | yes | yes |  |
| 27.2 | VGRD 75 mb | yes | yes |  |
| 28 | TKE 75 mb | yes | yes |  |
| 29 | HGT 100 mb | yes | yes |  |
| 30 | TMP 100 mb | yes | yes |  |
| 31 | RH 100 mb | yes | yes |  |
| 32 | VVEL 100 mb | yes | yes |  |
| 33 | DZDT 100 mb | yes | yes |  |
| 34.1 | UGRD 100 mb | yes | yes |  |
| 34.2 | VGRD 100 mb | yes | yes |  |
| 35 | TKE 100 mb | yes | yes |  |
| 36 | HGT 125 mb | yes | yes |  |
| 37 | TMP 125 mb | yes | yes |  |
| 38 | RH 125 mb | yes | yes |  |
| 39 | VVEL 125 mb | yes | yes |  |
| 40 | DZDT 125 mb | yes | yes |  |
| 41.1 | UGRD 125 mb | yes | yes |  |
| 41.2 | VGRD 125 mb | yes | yes |  |
| 42 | TKE 125 mb | yes | yes |  |
| 43 | HGT 150 mb | yes | yes |  |
| 44 | TMP 150 mb | yes | yes |  |
| 45 | RH 150 mb | yes | yes |  |
| 46 | VVEL 150 mb | yes | yes |  |
| 47 | DZDT 150 mb | yes | yes |  |
| 48.1 | UGRD 150 mb | yes | yes |  |
| 48.2 | VGRD 150 mb | yes | yes |  |
| 49 | TKE 150 mb | yes | yes |  |
| 50 | HGT 175 mb | yes | yes |  |
| 51 | TMP 175 mb | yes | yes |  |
| 52 | RH 175 mb | yes | yes |  |
| 53 | VVEL 175 mb | yes | yes |  |
| 54 | DZDT 175 mb | yes | yes |  |
| 55.1 | UGRD 175 mb | yes | yes |  |
| 55.2 | VGRD 175 mb | yes | yes |  |
| 56 | TKE 175 mb | yes | yes |  |
| 57 | HGT 200 mb | yes | yes |  |
| 58 | TMP 200 mb | yes | yes |  |
| 59 | RH 200 mb | yes | yes |  |
| 60 | VVEL 200 mb | yes | yes |  |
| 61 | DZDT 200 mb | yes | yes |  |
| 62.1 | UGRD 200 mb | yes | yes |  |
| 62.2 | VGRD 200 mb | yes | yes |  |
| 63 | TKE 200 mb | yes | yes |  |
| 64 | HGT 225 mb | yes | yes |  |
| 65 | TMP 225 mb | yes | yes |  |
| 66 | RH 225 mb | yes | yes |  |
| 67 | VVEL 225 mb | yes | yes |  |
| 68 | DZDT 225 mb | yes | yes |  |
| 69.1 | UGRD 225 mb | yes | yes |  |
| 69.2 | VGRD 225 mb | yes | yes |  |
| 70 | TKE 225 mb | yes | yes |  |
| 71 | HGT 250 mb | yes | yes |  |
| 72 | TMP 250 mb | yes | yes |  |
| 73 | RH 250 mb | yes | yes |  |
| 74 | VVEL 250 mb | yes | yes |  |
| 75 | DZDT 250 mb | yes | yes |  |
| 76.1 | UGRD 250 mb | yes | yes |  |
| 76.2 | VGRD 250 mb | yes | yes |  |
| 77 | ABSV 250 mb | yes | yes |  |
| 78 | TKE 250 mb | yes | yes |  |
| 79 | HGT 275 mb | yes | yes |  |
| 80 | TMP 275 mb | yes | yes |  |
| 81 | RH 275 mb | yes | yes |  |
| 82 | VVEL 275 mb | yes | yes |  |
| 83 | DZDT 275 mb | yes | yes |  |
| 84.1 | UGRD 275 mb | yes | yes |  |
| 84.2 | VGRD 275 mb | yes | yes |  |
| 85 | TKE 275 mb | yes | yes |  |
| 86 | HGT 300 mb | yes | yes |  |
| 87 | TMP 300 mb | yes | yes |  |
| 88 | RH 300 mb | yes | yes |  |
| 89 | VVEL 300 mb | yes | yes |  |
| 90 | DZDT 300 mb | yes | yes |  |
| 91.1 | UGRD 300 mb | yes | yes |  |
| 91.2 | VGRD 300 mb | yes | yes |  |
| 92 | TKE 300 mb | yes | yes |  |
| 93 | HGT 325 mb | yes | yes |  |
| 94 | TMP 325 mb | yes | yes |  |
| 95 | RH 325 mb | yes | yes |  |
| 96 | VVEL 325 mb | yes | yes |  |
| 97 | DZDT 325 mb | yes | yes |  |
| 98.1 | UGRD 325 mb | yes | yes |  |
| 98.2 | VGRD 325 mb | yes | yes |  |
| 99 | TKE 325 mb | yes | yes |  |
| 100 | HGT 350 mb | yes | yes |  |
| 101 | TMP 350 mb | yes | yes |  |
| 102 | RH 350 mb | yes | yes |  |
| 103 | VVEL 350 mb | yes | yes |  |
| 104 | DZDT 350 mb | yes | yes |  |
| 105.1 | UGRD 350 mb | yes | yes |  |
| 105.2 | VGRD 350 mb | yes | yes |  |
| 106 | TKE 350 mb | yes | yes |  |
| 107 | HGT 375 mb | yes | yes |  |
| 108 | TMP 375 mb | yes | yes |  |
| 109 | RH 375 mb | yes | yes |  |
| 110 | VVEL 375 mb | yes | yes |  |
| 111 | DZDT 375 mb | yes | yes |  |
| 112.1 | UGRD 375 mb | yes | yes |  |
| 112.2 | VGRD 375 mb | yes | yes |  |
| 113 | TKE 375 mb | yes | yes |  |
| 114 | HGT 400 mb | yes | yes |  |
| 115 | TMP 400 mb | yes | yes |  |
| 116 | RH 400 mb | yes | yes |  |
| 117 | VVEL 400 mb | yes | yes |  |
| 118 | DZDT 400 mb | yes | yes |  |
| 119.1 | UGRD 400 mb | yes | yes |  |
| 119.2 | VGRD 400 mb | yes | yes |  |
| 120 | TKE 400 mb | yes | yes |  |
| 121 | HGT 425 mb | yes | yes |  |
| 122 | TMP 425 mb | yes | yes |  |
| 123 | RH 425 mb | yes | yes |  |
| 124 | VVEL 425 mb | yes | yes |  |
| 125 | DZDT 425 mb | yes | yes |  |
| 126.1 | UGRD 425 mb | yes | yes |  |
| 126.2 | VGRD 425 mb | yes | yes |  |
| 127 | TKE 425 mb | yes | yes |  |
| 128 | HGT 450 mb | yes | yes |  |
| 129 | TMP 450 mb | yes | yes |  |
| 130 | RH 450 mb | yes | yes |  |
| 131 | VVEL 450 mb | yes | yes |  |
| 132 | DZDT 450 mb | yes | yes |  |
| 133.1 | UGRD 450 mb | yes | yes |  |
| 133.2 | VGRD 450 mb | yes | yes |  |
| 134 | TKE 450 mb | yes | yes |  |
| 135 | HGT 475 mb | yes | yes |  |
| 136 | TMP 475 mb | yes | yes |  |
| 137 | RH 475 mb | yes | yes |  |
| 138 | VVEL 475 mb | yes | yes |  |
| 139 | DZDT 475 mb | yes | yes |  |
| 140.1 | UGRD 475 mb | yes | yes |  |
| 140.2 | VGRD 475 mb | yes | yes |  |
| 141 | TKE 475 mb | yes | yes |  |
| 142 | HGT 500 mb | yes | yes |  |
| 143 | TMP 500 mb | yes | yes |  |
| 144 | RH 500 mb | yes | yes |  |
| 145 | VVEL 500 mb | yes | yes |  |
| 146 | DZDT 500 mb | yes | yes |  |
| 147.1 | UGRD 500 mb | yes | yes |  |
| 147.2 | VGRD 500 mb | yes | yes |  |
| 148 | ABSV 500 mb | yes | yes |  |
| 149 | TKE 500 mb | yes | yes |  |
| 150 | HGT 525 mb | yes | yes |  |
| 151 | TMP 525 mb | yes | yes |  |
| 152 | RH 525 mb | yes | yes |  |
| 153 | VVEL 525 mb | yes | yes |  |
| 154 | DZDT 525 mb | yes | yes |  |
| 155.1 | UGRD 525 mb | yes | yes |  |
| 155.2 | VGRD 525 mb | yes | yes |  |
| 156 | TKE 525 mb | yes | yes |  |
| 157 | HGT 550 mb | yes | yes |  |
| 158 | TMP 550 mb | yes | yes |  |
| 159 | RH 550 mb | yes | yes |  |
| 160 | VVEL 550 mb | yes | yes |  |
| 161 | DZDT 550 mb | yes | yes |  |
| 162.1 | UGRD 550 mb | yes | yes |  |
| 162.2 | VGRD 550 mb | yes | yes |  |
| 163 | TKE 550 mb | yes | yes |  |
| 164 | HGT 575 mb | yes | yes |  |
| 165 | TMP 575 mb | yes | yes |  |
| 166 | RH 575 mb | yes | yes |  |
| 167 | VVEL 575 mb | yes | yes |  |
| 168 | DZDT 575 mb | yes | yes |  |
| 169.1 | UGRD 575 mb | yes | yes |  |
| 169.2 | VGRD 575 mb | yes | yes |  |
| 170 | TKE 575 mb | yes | yes |  |
| 171 | HGT 600 mb | yes | yes |  |
| 172 | TMP 600 mb | yes | yes |  |
| 173 | RH 600 mb | yes | yes |  |
| 174 | VVEL 600 mb | yes | yes |  |
| 175 | DZDT 600 mb | yes | yes |  |
| 176.1 | UGRD 600 mb | yes | yes |  |
| 176.2 | VGRD 600 mb | yes | yes |  |
| 177 | TKE 600 mb | yes | yes |  |
| 178 | HGT 625 mb | yes | yes |  |
| 179 | TMP 625 mb | yes | yes |  |
| 180 | RH 625 mb | yes | yes |  |
| 181 | VVEL 625 mb | yes | yes |  |
| 182 | DZDT 625 mb | yes | yes |  |
| 183.1 | UGRD 625 mb | yes | yes |  |
| 183.2 | VGRD 625 mb | yes | yes |  |
| 184 | TKE 625 mb | yes | yes |  |
| 185 | HGT 650 mb | yes | yes |  |
| 186 | TMP 650 mb | yes | yes |  |
| 187 | RH 650 mb | yes | yes |  |
| 188 | VVEL 650 mb | yes | yes |  |
| 189 | DZDT 650 mb | yes | yes |  |
| 190.1 | UGRD 650 mb | yes | yes |  |
| 190.2 | VGRD 650 mb | yes | yes |  |
| 191 | TKE 650 mb | yes | yes |  |
| 192 | HGT 675 mb | yes | yes |  |
| 193 | TMP 675 mb | yes | yes |  |
| 194 | RH 675 mb | yes | yes |  |
| 195 | VVEL 675 mb | yes | yes |  |
| 196 | DZDT 675 mb | yes | yes |  |
| 197.1 | UGRD 675 mb | yes | yes |  |
| 197.2 | VGRD 675 mb | yes | yes |  |
| 198 | TKE 675 mb | yes | yes |  |
| 199 | HGT 700 mb | yes | yes |  |
| 200 | TMP 700 mb | yes | yes |  |
| 201 | RH 700 mb | yes | yes |  |
| 202 | VVEL 700 mb | yes | yes |  |
| 203 | DZDT 700 mb | yes | yes |  |
| 204.1 | UGRD 700 mb | yes | yes |  |
| 204.2 | VGRD 700 mb | yes | yes |  |
| 205 | ABSV 700 mb | yes | yes |  |
| 206 | TKE 700 mb | yes | yes |  |
| 207 | HGT 725 mb | yes | yes |  |
| 208 | TMP 725 mb | yes | yes |  |
| 209 | RH 725 mb | yes | yes |  |
| 210 | VVEL 725 mb | yes | yes |  |
| 211 | DZDT 725 mb | yes | yes |  |
| 212.1 | UGRD 725 mb | yes | yes |  |
| 212.2 | VGRD 725 mb | yes | yes |  |
| 213 | TKE 725 mb | yes | yes |  |
| 214 | HGT 750 mb | yes | yes |  |
| 215 | TMP 750 mb | yes | yes |  |
| 216 | RH 750 mb | yes | yes |  |
| 217 | VVEL 750 mb | yes | yes |  |
| 218 | DZDT 750 mb | yes | yes |  |
| 219.1 | UGRD 750 mb | yes | yes |  |
| 219.2 | VGRD 750 mb | yes | yes |  |
| 220 | TKE 750 mb | yes | yes |  |
| 221 | HGT 775 mb | yes | yes |  |
| 222 | TMP 775 mb | yes | yes |  |
| 223 | RH 775 mb | yes | yes |  |
| 224 | VVEL 775 mb | yes | yes |  |
| 225 | DZDT 775 mb | yes | yes |  |
| 226.1 | UGRD 775 mb | yes | yes |  |
| 226.2 | VGRD 775 mb | yes | yes |  |
| 227 | TKE 775 mb | yes | yes |  |
| 228 | HGT 800 mb | yes | yes |  |
| 229 | TMP 800 mb | yes | yes |  |
| 230 | RH 800 mb | yes | yes |  |
| 231 | VVEL 800 mb | yes | yes |  |
| 232 | DZDT 800 mb | yes | yes |  |
| 233.1 | UGRD 800 mb | yes | yes |  |
| 233.2 | VGRD 800 mb | yes | yes |  |
| 234 | TKE 800 mb | yes | yes |  |
| 235 | HGT 825 mb | yes | yes |  |
| 236 | TMP 825 mb | yes | yes |  |
| 237 | RH 825 mb | yes | yes |  |
| 238 | VVEL 825 mb | yes | yes |  |
| 239 | DZDT 825 mb | yes | yes |  |
| 240.1 | UGRD 825 mb | yes | yes |  |
| 240.2 | VGRD 825 mb | yes | yes |  |
| 241 | TKE 825 mb | yes | yes |  |
| 242 | HGT 850 mb | yes | yes |  |
| 243 | TMP 850 mb | yes | yes |  |
| 244 | RH 850 mb | yes | yes |  |
| 245 | VVEL 850 mb | yes | yes |  |
| 246 | DZDT 850 mb | yes | yes |  |
| 247.1 | UGRD 850 mb | yes | yes |  |
| 247.2 | VGRD 850 mb | yes | yes |  |
| 248 | ABSV 850 mb | yes | yes |  |
| 249 | TKE 850 mb | yes | yes |  |
| 250 | HGT 875 mb | yes | yes |  |
| 251 | TMP 875 mb | yes | yes |  |
| 252 | RH 875 mb | yes | yes |  |
| 253 | VVEL 875 mb | yes | yes |  |
| 254 | DZDT 875 mb | yes | yes |  |
| 255.1 | UGRD 875 mb | yes | yes |  |
| 255.2 | VGRD 875 mb | yes | yes |  |
| 256 | TKE 875 mb | yes | yes |  |
| 257 | HGT 900 mb | yes | yes |  |
| 258 | TMP 900 mb | yes | yes |  |
| 259 | RH 900 mb | yes | yes |  |
| 260 | VVEL 900 mb | yes | yes |  |
| 261 | DZDT 900 mb | yes | yes |  |
| 262.1 | UGRD 900 mb | yes | yes |  |
| 262.2 | VGRD 900 mb | yes | yes |  |
| 263 | TKE 900 mb | yes | yes |  |
| 264 | HGT 925 mb | yes | yes |  |
| 265 | TMP 925 mb | yes | yes |  |
| 266 | RH 925 mb | yes | yes |  |
| 267 | VVEL 925 mb | yes | yes |  |
| 268 | DZDT 925 mb | yes | yes |  |
| 269.1 | UGRD 925 mb | yes | yes |  |
| 269.2 | VGRD 925 mb | yes | yes |  |
| 270 | TKE 925 mb | yes | yes |  |
| 271 | HGT 950 mb | yes | yes |  |
| 272 | TMP 950 mb | yes | yes |  |
| 273 | RH 950 mb | yes | yes |  |
| 274 | VVEL 950 mb | yes | yes |  |
| 275 | DZDT 950 mb | yes | yes |  |
| 276.1 | UGRD 950 mb | yes | yes |  |
| 276.2 | VGRD 950 mb | yes | yes |  |
| 277 | TKE 950 mb | yes | yes |  |
| 278 | HINDEX surface | yes | yes |  |
| 279 | HGT 975 mb | yes | yes |  |
| 280 | TMP 975 mb | yes | yes |  |
| 281 | RH 975 mb | yes | yes |  |
| 282 | VVEL 975 mb | yes | yes |  |
| 283 | DZDT 975 mb | yes | yes |  |
| 284.1 | UGRD 975 mb | yes | yes |  |
| 284.2 | VGRD 975 mb | yes | yes |  |
| 285 | TKE 975 mb | yes | yes |  |
| 286 | TMP 1000 mb | yes | yes |  |
| 287 | RH 1000 mb | yes | yes |  |
| 288 | VVEL 1000 mb | yes | yes |  |
| 289 | DZDT 1000 mb | yes | yes |  |
| 290.1 | UGRD 1000 mb | yes | yes |  |
| 290.2 | VGRD 1000 mb | yes | yes |  |
| 291 | ABSV 1000 mb | yes | yes |  |
| 292 | TKE 1000 mb | yes | yes |  |
| 293 | MSLET mean sea level | yes | yes |  |
| 294 | HGT 1000 mb | yes | yes |  |
| 295 | REFD 4000 m above ground | yes | yes |  |
| 296 | REFD 1000 m above ground | yes | yes |  |
| 297 | PRES surface | yes | yes |  |
| 298 | HGT surface | yes | yes |  |
| 299 | TMP surface | yes | yes |  |
| 300 | TSOIL 0-0.1 m below ground | yes | yes |  |
| 301 | SOILW 0-0.1 m below ground | yes | yes |  |
| 302 | SOILL 0-0.1 m below ground | yes | yes |  |
| 303 | TSOIL 0.1-0.4 m below ground | yes | yes |  |
| 304 | SOILW 0.1-0.4 m below ground | yes | yes |  |
| 305 | SOILL 0.1-0.4 m below ground | yes | yes |  |
| 306 | TSOIL 0.4-1 m below ground | yes | yes |  |
| 307 | SOILW 0.4-1 m below ground | yes | yes |  |
| 308 | SOILL 0.4-1 m below ground | yes | yes |  |
| 309 | TSOIL 1-2 m below ground | yes | yes |  |
| 310 | SOILW 1-2 m below ground | yes | yes |  |
| 311 | SOILL 1-2 m below ground | yes | yes |  |
| 312 | TSOIL 3 m underground | yes | yes |  |
| 313 | MSTAV 0-1 m below ground | yes | yes |  |
| 314 | SOILM 0-2 m below ground | yes | yes |  |
| 315 | CNWAT surface | yes | yes |  |
| 316 | WEASD surface | yes | yes |  |
| 317 | SNOWC surface | yes | yes |  |
| 318 | SNOD surface | yes | yes |  |
| 319 | SMDRY surface | yes | yes |  |
| 320 | POROS surface | yes | yes |  |
| 321 | TMP 2 m above ground | yes | yes |  |
| 322 | SPFH 2 m above ground | yes | yes |  |
| 323 | DPT 2 m above ground | yes | yes |  |
| 324 | RH 2 m above ground | yes | yes |  |
| 325 | TMAX 2 m above ground (5-6 h max) | yes | yes |  |
| 326 | TMIN 2 m above ground (5-6 h min) | yes | yes |  |
| 327 | MAXRH 2 m above ground (5-6 h max) | yes | yes |  |
| 328 | MINRH 2 m above ground (5-6 h min) | yes | yes |  |
| 329.1 | UGRD 10 m above ground | yes | yes |  |
| 329.2 | VGRD 10 m above ground | yes | yes |  |
| 330 | MAXUW 10 m above ground (5-6 h max) | no | - | not in the checked file: its history predates the hourly 10 m maximum wind output (see COMPATIBILITY.md) |
| 331 | MAXVW 10 m above ground (5-6 h max) | no | - | not in the checked file: its history predates the hourly 10 m maximum wind output (see COMPATIBILITY.md) |
| 332 | APCP surface (0-6 h acc) | yes | yes |  |
| 333 | ACPCP surface (0-6 h acc) | yes | yes |  |
| 334 | WEASD surface (0-6 h acc) | no | - | not produced (see COMPATIBILITY.md, 'Not in the files') |
| 335 | HGT lowest level of the wet bulb zero | yes | yes |  |
| 336 | CSNOW surface | no | - | not produced (see COMPATIBILITY.md, 'Not in the files') |
| 337 | CICEP surface | no | - | not produced (see COMPATIBILITY.md, 'Not in the files') |
| 338 | CFRZR surface | no | - | not produced (see COMPATIBILITY.md, 'Not in the files') |
| 339 | CRAIN surface | no | - | not produced (see COMPATIBILITY.md, 'Not in the files') |
| 340 | SFCR surface | yes | yes |  |
| 341 | FRICV surface | yes | yes |  |
| 342 | CD surface | yes | yes |  |
| 343 | VEG surface | yes | yes |  |
| 344 | VGTYP surface | yes | yes |  |
| 345 | SOTYP surface | yes | yes |  |
| 346 | RLYRS surface | yes | yes |  |
| 347 | WILT surface | yes | yes |  |
| 348 | SMREF surface | yes | yes |  |
| 349 | LFTX 500-1000 mb | yes | yes |  |
| 350 | CAPE surface | yes | yes |  |
| 351 | CIN surface | yes | yes |  |
| 352 | PWAT entire atmosphere (considered as a single layer) | yes | yes |  |
| 353 | HGT cloud base | yes | yes |  |
| 354 | HLCY 3000-0 m above ground | yes | yes |  |
| 355 | VUCSH 0-6000 m above ground | yes | yes |  |
| 356 | VVCSH 0-6000 m above ground | yes | yes |  |
| 357 | PRES tropopause | yes | yes |  |
| 358 | TMP tropopause | yes | yes |  |
| 359.1 | UGRD tropopause | yes | yes |  |
| 359.2 | VGRD tropopause | yes | yes |  |
| 360 | PRES max wind | yes | yes |  |
| 361.1 | UGRD max wind | yes | yes |  |
| 361.2 | VGRD max wind | yes | yes |  |
| 362 | TMP 80 m above ground | yes | yes |  |
| 363 | SPFH 80 m above ground | yes | yes |  |
| 364 | PRES 80 m above ground | yes | yes |  |
| 365.1 | UGRD 80 m above ground | yes | yes |  |
| 365.2 | VGRD 80 m above ground | yes | yes |  |
| 366 | HGT 0C isotherm | yes | yes |  |
| 367 | RH 0C isotherm | yes | yes |  |
| 368 | TMP 30-0 mb above ground | yes | yes |  |
| 369 | RH 30-0 mb above ground | yes | yes |  |
| 370.1 | UGRD 30-0 mb above ground | yes | yes |  |
| 370.2 | VGRD 30-0 mb above ground | yes | yes |  |
| 371 | PLI 30-0 mb above ground | yes | yes |  |
| 372 | TMP 60-30 mb above ground | yes | yes |  |
| 373 | RH 60-30 mb above ground | yes | yes |  |
| 374.1 | UGRD 60-30 mb above ground | yes | yes |  |
| 374.2 | VGRD 60-30 mb above ground | yes | yes |  |
| 375 | TMP 90-60 mb above ground | yes | yes |  |
| 376 | RH 90-60 mb above ground | yes | yes |  |
| 377.1 | UGRD 90-60 mb above ground | yes | yes |  |
| 377.2 | VGRD 90-60 mb above ground | yes | yes |  |
| 378 | TMP 120-90 mb above ground | yes | yes |  |
| 379 | RH 120-90 mb above ground | yes | yes |  |
| 380.1 | UGRD 120-90 mb above ground | yes | yes |  |
| 380.2 | VGRD 120-90 mb above ground | yes | yes |  |
| 381 | TMP 150-120 mb above ground | yes | yes |  |
| 382 | RH 150-120 mb above ground | yes | yes |  |
| 383.1 | UGRD 150-120 mb above ground | yes | yes |  |
| 383.2 | VGRD 150-120 mb above ground | yes | yes |  |
| 384 | TMP 180-150 mb above ground | yes | yes |  |
| 385 | RH 180-150 mb above ground | yes | yes |  |
| 386.1 | UGRD 180-150 mb above ground | yes | yes |  |
| 386.2 | VGRD 180-150 mb above ground | yes | yes |  |
| 387 | 4LFTX 180-0 mb above ground | yes | yes |  |
| 388 | CAPE 180-0 mb above ground | yes | yes |  |
| 389 | CIN 180-0 mb above ground | yes | yes |  |
| 390 | HPBL surface | yes | yes |  |
| 391 | RH 0.33-1 sigma layer | yes | yes |  |
| 392 | LTNG surface | no | - | not produced (see COMPATIBILITY.md, 'Not in the files') |
| 393 | LAND surface | yes | yes |  |
| 394 | ICEC surface | yes | yes |  |
| 395 | ALBDO surface | yes | yes |  |
| 396 | SHTFL surface | yes | yes |  |
| 397 | LHTFL surface | yes | yes |  |
| 398 | TCDC entire atmosphere (considered as a single layer) | yes | yes |  |
| 399 | DSWRF surface | yes | yes |  |
| 400 | DLWRF surface | yes | yes |  |
| 401 | USWRF surface | yes | yes |  |
| 402 | ULWRF surface | yes | yes |  |
| 403 | APCP surface (3-6 h acc) | yes | yes |  |
| 404 | ACPCP surface (3-6 h acc) | yes | yes |  |
