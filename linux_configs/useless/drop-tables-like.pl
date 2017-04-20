#!/usr/bin/perl

use strict;
use warnings;

use Yabs::CollectBase;
use Yabs::Logger;

my $table_like = shift(@ARGV);

my $dbh = connect_local()
or exit ERROR "connect_local failed";

my $tables = $dbh->selectcol_arrayref("
	show tables like '$table_like'
")
or exit ERROR "selectcol_arrayref failed: " . $dbh->errstr();

unless (scalar(@$tables)) {
	INFO "No tables found, exit";
	exit(0);
}

INFO "Tables are : \n" . join("\n", @$tables) . "\n\n Drop?";
my $symbol = <>;

chomp($symbol);

if ($symbol eq 'y') {
	$dbh->do("
		drop tables " . join(", ", @$tables) . "
	")
	or exit ERROR "drop tables failed : " . $dbh->errstr();

	INFO "Dropped OK";
}

