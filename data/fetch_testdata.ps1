# Fetch testdata from datasets hosted by TU Wien.
#
# https://www.cg.tuwien.ac.at/research/vis/datasets/

$zippaths = "https://www.cg.tuwien.ac.at/research/publications/2006/dataset-present/dataset-present-246x246x221.zip",
		 	"https://www.cg.tuwien.ac.at/research/publications/2006/dataset-present/dataset-present-328x328x294.zip",
		 	"https://www.cg.tuwien.ac.at/research/publications/2006/dataset-present/dataset-present-492x492x442.zip",
		 	"https://www.cg.tuwien.ac.at/research/publications/2005/dataset-stagbeetle/dataset-stagbeetle-208x208x123.zip",
		 	"https://www.cg.tuwien.ac.at/research/publications/2005/dataset-stagbeetle/dataset-stagbeetle-277x277x164.zip",
		 	"https://www.cg.tuwien.ac.at/research/publications/2005/dataset-stagbeetle/dataset-stagbeetle-832x832x494.zip",
		 	"https://www.cg.tuwien.ac.at/research/publications/2002/dataset-christmastree/dataset-christmastree-128x124x128.zip",
		 	"https://www.cg.tuwien.ac.at/research/publications/2002/dataset-christmastree/dataset-christmastree-170x166x170.zip",
		 	"https://www.cg.tuwien.ac.at/research/publications/2002/dataset-christmastree/dataset-christmastree-256x249x256.zip",
		 	"https://www.cg.tuwien.ac.at/research/publications/2002/dataset-christmastree/dataset-christmastree-512x499x512.zip"

Foreach ($path in $zippaths)
{
	Invoke-Webrequest $path -o tmp.zip
	Expand-Archive -Force -DestinationPath .\ tmp.zip
	Remove-Item tmp.zip
}

Invoke-Webrequest https://www.cg.tuwien.ac.at/research/publications/2005/dataset-stagbeetle/dataset-stagbeetle-416x416x247.dat -o dataset-stagbeetle-416x416x247.dat
