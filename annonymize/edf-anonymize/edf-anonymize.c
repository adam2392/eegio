/* file: edf-anonymize.c	G. Moody	28 April 2010

-------------------------------------------------------------------------------
edf-anonymize: Make an anonymized copy of an EDF/EDF+ file
Copyright (C) 2010 George B. Moody

This program is free software; you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation; either version 2 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
this program; if not, write to the Free Software Foundation, Inc., 59 Temple
Place - Suite 330, Boston, MA 02111-1307, USA.

You may contact the author by e-mail (george@mit.edu) or postal mail
(MIT Room E25-505A, Cambridge, MA 02139 USA).  For updates to this software,
please visit PhysioNet (http://www.physionet.org/).
_______________________________________________________________________________

Anonymization of an EDF/EDF+ file requires removal of the patient name and
id, and all elements of the recording date other than the year.  These can
be replaced with surrogate data, and this is recommended since some software
intended to read EDF/EDF+ files may reject inputs in which the fields for
name, id, and recording date are empty.

By default, however, this program replaces the name and id with space
(blank) characters, and it replaces the recording day and month with '01.01'
(1 January) without altering the recording year.

Run the program from the command line without any arguments to obtain a brief
synopsis of its use.  A few examples:

   edf-anonymize foo.edf anonymous.edf
      This copies the contents of foo.edf to a new file called anonymous.edf,
      emptying the patient name and id fields and setting the date to 01.01
      without altering the recording year.

   edf-anonymize foo.edf anonymous.edf "Arthur Dent"
      As above, but the patient's name is replaced by Arthur Dent.  The
      quotation marks are needed if the name contains any spaces, as in this
      example. Names longer than 80 characters are truncated. 

   edf-anonymize foo.edf anonymous.edf "Arthur Dent" 42
      As above, but the patient id is replaced by 42. Ids longer than 80
      characters are truncated.

   edf-anonymize foo.edf anonymous.edf "Arthur Dent" 42 03.04.05 
      As above, but the date is replaced by 03.04.05 (i.e, 3 April 2005).

   edf-anonymize foo.edf anonymous.edf "Arthur Dent" 42 03.04
      As above, but the day and month are replaced by 03.04, without altering
      the recording year.

   edf-anonymize foo.edf anonymous.edf "Arthur Dent" 42 +12345
      As above, but the recording date is shifted 12345 days into the future.

   edf-anonymize foo.edf anonymous.edf "Arthur Dent" 42 +-5678
      As above, but the recording date is shifted -5678 days into the past.
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Convert a date string in the form DD.MM.YY (as specified by the EDF
   standard) to a Julian date between 1 Jan 1985 and 31 Dec 2084 (the
   range of dates specified by the EDF standard).  Yes, this is overkill
   for this application! */
long strtojul(char *string)
{
    char *mp, *yp;
    int d, m, y, gcorr, jm, jy;
    long date;

    if ((mp = strchr(string,'.')) == NULL || (yp = strchr(mp+1,'.')) == NULL ||
	(d = atoi(string)) < 1 || d > 31 || (m = atoi(mp+1)) < 1 || m > 12)
	return (0L);
    if ((y = atoi(yp+1) + 1900) < 1985) y += 100;
    if (m > 2) { jy = y; jm = m + 1; }
    else { jy = y - 1; 	jm = m + 13; }
    if (jy > 0) date = 365.25*jy;
    else date = -(long)(-365.25 * (jy + 0.25));
    date += (int)(30.6001*jm) + d + 1720995L;
    if (d + 31L*(m + 12L*y) >= (15 + 31L*(10 + 12L*1582))) { /* 15/10/1582 */
	gcorr = (int)(0.01*jy);		/* Gregorian calendar correction */
	date += 2 - gcorr + (int)(0.25*gcorr);
    }
    return (date);
}

char date_string[10];

/* Convert a Julian date to a date string in the form DD.MM.YY */
char *jultostr(long date)
{
    int d, m, y, gcorr, jm, jy;
    long jd;

    if (date >= 2299161L) {	/* Gregorian calendar correction */
	gcorr = (int)(((date - 1867216L) - 0.25)/36524.25);
	date += 1 + gcorr - (long)(0.25*gcorr);
    }
    date += 1524;
    jy = (int)(6680 + ((date - 2439870L) - 122.1)/365.25);
    jd = 365L*jy + (0.25*jy);
    jm = (int)((date - jd)/30.6001);
    d = date - jd - (int)(30.6001*jm);
    if ((m = jm - 1) > 12) m -= 12;
    y = jy - 4715;
    if (m > 2) y--;
    if (y <= 0) y--;
    y %= 100;
    (void)sprintf(date_string, "%02d.%02d.%02d", d, m, y);
    return (date_string);
}

main(int argc, char **argv)
{
    static char buf[1024];
    FILE *ifile, *ofile;
    int i, n;

    if (argc < 3 || strcmp(argv[1], argv[2]) == 0) {
	fprintf(stderr, "usage: %s INPUT OUTPUT [SNAME [SID [SDATE]]]\n"
  " where INPUT is the name of the EDF file to be anonymized,\n"
  " OUTPUT is the name of the anonymized EDF file to be written,\n"
  " SNAME and SID are the surrogate name and ID to be written into OUTPUT,\n"
  " and SDATE is the surrogate date to be written, in the form DD.MM.YY\n"
  "   (or DD.MM, or +DAYS)\n"
  " SNAME and SID are optional and empty by default.\n"
  " SDATE is optional, and is '01.01' (1 January) by default.\n",
		argv[0]);
	exit(1);
    }
    if ((ifile = fopen(argv[1], "rb")) == NULL) {
	fprintf(stderr, "%s: can't open %s\n", argv[0], argv[1]);
	exit(2);
    }
    /* All EDF/EDF+ files begin with a 256-byte fixed-format header */
    n = fread(buf, 1, sizeof(buf), ifile);
    if (n < 256) {
	fprintf(stderr, "%s: %s is too short for an EDF/EDF+ file\n"
		" No output written\n",
		argv[0], argv[1]);
	fclose(ifile);
	exit(3);
    }

    /* Replace name with surrogate name. */
    i = 0;
    if (argc > 3)
	for (i = 0; i < 80 && argv[3][i]; i++)
	    buf[i+8] = argv[3][i];
    for ( ; i < 80; i++)
	buf[i+8] = ' ';

    /* Replace id with surrogate id. */
    i = 0;
    if (argc > 4)
	for (i = 0; i < 80 && argv[4][i]; i++)
	    buf[i+88] = argv[4][i];
    for ( ; i < 80; i++)
	buf[i+88] = ' ';

    /* Replace date with surrogate date. */
    if (argc <= 5)
	strncpy(buf+168, "01.01", 5);  /* 01.01 is 1 January */
    else if (argv[5][0] == '+') {
	static char date[9];
	long delta, jdate;

	sscanf(argv[5]+1, "%ld", &delta);
	strncpy(date, buf+168, 8);
	jdate = strtojul(date) + delta;
	strncpy(buf+168, jultostr(jdate), 8);
    }
    else if (strlen(argv[5]) == 7 && argv[5][2] == '.' && argv[5][5] == '.')
	strncpy(buf+168, argv[5], 7);
    else if (strlen(argv[5]) == 5 && argv[5][2] == '.')
	strncpy(buf+168, argv[5], 5);
    else {
	fprintf(stderr,
		"%s: improper format (%s) for surrogate date\n",
		argv[0], argv[5]);
	fclose(ifile);
	exit(4);
    }

    if ((ofile = fopen(argv[2], "wb")) == NULL) {
	fprintf(stderr, "%s: can't open %s\n", argv[0], argv[2]);
	fclose(ifile);
	exit(5);
    }

    /* Check the format and warn if the reserved area is not empty. */
    if (strncmp("EDF+C", buf+192, 5) == 0) {
	fprintf(stderr, "Format: EDF+C\n");
	i = 5;
    }
    else if (strncmp("EDF+D", buf+192, 5) == 0) {
	fprintf(stderr, "Format: EDF+D\n");
	i = 5;
    }
    else {
	fprintf(stderr, "Format: EDF\n");
	i = 0;
    }
    for ( ; i < 44; i++) {
	if (buf[i+192] != ' ') {
	    fprintf(stderr, "%s: WARNING\n"
		    " Reserved area of header is not empty as expected\n"
		    " Check for possible PHI in bytes %d-236 of %s\n",
		    argv[0], i+192, argv[2]);
	    break;
	}
    }

    /* Write the output, beginning with the anonymized header and copying
       the remainder of the input. */
    do 
	fwrite(buf, 1, n, ofile);
    while (n = fread(buf, 1, 1024, ifile));

    fclose(ofile);
    fclose(ifile);
    exit(0);
}
