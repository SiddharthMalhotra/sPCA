##############################################################
## This script normalizes tweets including elonged words by ##
## reducing them to standard length                         ##
## the orginal length.                                      ##
## Usage: cat text | perl Resolve.pl > normalized           ##
## Walid Magdy - QCRI                                       ##
## Created: 26/2/2012                                       ##
## Last modified: 18/6/2013                                 ##
##############################################################

use utf8;
binmode STDIN, ':utf8';
binmode STDOUT, ':utf8';

$dicFile = "JSC.pruned.dic" or die "Please specify a dictionary of correct words\n";
$elongFile = "Elong.class.txt" or die "Please specify the elonged class file\n";

open(IN1, $dicFile) or die "Can't find $dicFile\n";
binmode IN1, ':utf8';
open(IN2, $elongFile) or die "Can't find $elongFile\n";
binmode IN2, ':utf8';

## Reading Al-Jazeera Dictionary (The MSA reference dictionary)
while(<IN1>){
  if(/(\S\S+)\s+(\d\d+)/){
    $dic{$1} = $2;
  }
}
close(IN1);

## Reading the classes of Elonged words for normalizing elongaions in tweets
$MostFreq = 1;
while(<IN2>){
  if(/----------/){
    $MostFreq = 1;
  }
  elsif($MostFreq && /(\S+)\s+\d+/){
    $word = $base = $1;
    $base =~ s/(\S)\1+/\1/g;
    $Norm{$base}=$word;
    $MostFreq = 0;
  }
}
close(IN2);

## Reading Tweets
while(<STDIN>){
	chomp;
	$tweet = $_;
	my @chunks = split '\t', $_;
	$id = $chunks[2];
	$tweet = $chunks[5];
	$tweet =~ s/^RT //g;     ## Removing retweet headers
	$tweet = &normalize($tweet);   ## normalizing tweets
	$normalized = "";
    while($tweet =~ /(\S+)/g){
      $normalized .= &shorten($1)." ";   ## Resolving elongations
    }
	print $id,"\t",$normalized,"\n";
}

sub normalize(){
	my $txt = shift;
	$txt =~ s/[ًٌٍَُِّـْ]//g;     ## removing diacritics and kashidas
	$txt =~ tr/أآإىة/ااايه/;    ## normalizing standard characters
	## normalizing Farsi characters
	$txt =~ s/[ﻻﻵﻷעﻹﻼﻶﻸﬠ]/لا/g;
	$txt =~tr/٠١٢۲٣۳٤٥Ƽ٦٧۷٨۸٩۹ﺀٴٱﺂﭑﺎﺈﺄιٲﺍٳίﺃٵﺇﺁﺑپﮨﺒٻﺐﺏﭘﭒﭗﭖﭚٮﭛٺﺗﭠﺘټﺖﺕﭡٹﭞٿﭟﭤﮢﭥﭨﭢﭣﮣﭧﺛﺜﮆﺚﺙٽﮇچﺟﭴﺠﭼڄڇﭸﺝڃﺞﭽﮀﭵﭹﭻﭾﭿﭺﺣﺤﺡﺢځﺧﺨڅڂﺦﺥڿډﺩڍﺪڊڈﮃﮂڋﮈڌﮉڐﮄﺫﺬڎڏۮڕړﺮﺭڒڔږڑژﮌڗﮍڙﺯﺰﮊﺳڛﺴﺲﺱښﺷڜﺸﺶﺵۺﺻﺼڝﺺﺹﺿﻀﺽڞﺾۻﻃﻁﻄﻂﻈﻇﻅڟﻆﻋ۶ﻌﻊﻉﻏﻐڠۼﻍﻎﻓڤﻔﭬڣﭰﻒﻑڦڢڡﭫڥﭪﭭﭯﭮﻗﻘڨﻖﻕڧﭱگڳکڪڱﮔﻛﮘڰﮐﮖﻜﮜڲﻚڴﮗڭﻙﮓﮙګڮﮕﮛڬﮎﮝﮚﮑﮒﮏﯖﯕﻟڵڷﻠڶﻞﻝڸﻣﻤﻢﻡﻧﻥڼﻨﻦڻڽﮠڹﮞںטּﮡﮟھہۃﮬﮪﮧۂﻫﮫﺔﻪﻬﮭﺓۿﻩەۀﮤﮥﮦۆۈۅﯙۉﻭﻮۄۋۇۊﯚٷٶﯛﯠﺆﯜۏﺅﯡﯝﯘﯢﯞﯣﯗﯟﯾےﻳۓېێﮱﻴﮯﭔﻲۑۍﯿﻱﻰﭜڀﺋﻯﭕﮮﺌﭓﯼﭝ༦ﺊﯽﮰﭙﯥﺉﯦﯧﯤیٸ/0122334556778899ءءاااااااااااااااببببببببببببببتتتتتتتتتتتتتتتتتتتتثثثثثثثجججججججججججججججججججحححححخخخخخخخددددددددددددددذذذذذرررررررررررررزززسسسسسسششششششصصصصصضضضضضضططططظظظظظعععععغغغغغغفففففففففففففففففقققققققككككككككككككككككككككككككككككككككككللللللللممممننننننننننننننهههههههههههههههههههههوووووووووووووووووووووووووووويييييييييييييييييييييييييييييييييييييي/;
	@URL = ();
	$n=0;
	while($txt =~ s/(http\S+)/URL$n/){
		$URL[$n++] = $1;
	}
	$txt =~ s/\:\-*[\)D]+/ LOL /g;
	$txt =~ s/\^[\_]+\^/ LOL /g;
	$txt =~ s/\:[\-\']*\(+/ SAD /g;
	$txt =~ s/[^اأإآبتثجحخدذرزسشصضطظعغفقكلمنهويىئءؤةa-zA-Z0-9\@\#\_]+/ /g;  ## Filtering noice and puncituations in text
	## to be added: \-\(\)\:\;\^
	$txt =~ s/ l+o+l+ / LOL /gi;
	$txt =~ s/ ل+و+ل+ / LOL /g;
	$txt =~ s/ LOL / /g;
	for ($i=0; $i<=$#URL; $i++){
		$txt =~ s/URL$i/$URL[$i]/;
	}
	return $txt;
}

sub shorten(){
  $word = $base = shift;
  if($word =~ /[^ابتثجحخدذرزسشصضطظعغفقكلمنهويآإءىةئؤ]/){
    return $word;
  }
  else{
    if(exists $dic{$word}){
      $norm = $word;
    }
    else{
      $base =~ s/(\S)\1+/\1/g;
      if(exists $Norm{$base}){
	$norm = $Norm{$base};
      }
      else{
	$norm = "$base";
      }
    }
    return $norm;
  }
}
