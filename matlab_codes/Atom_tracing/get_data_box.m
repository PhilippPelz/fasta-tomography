function DataBox = get_data_box(CurrData,x,y,z,BoxRadius)
%M. Bartels, UCLA, 2014
%small function to extract Databox of a specifix radios from a Density
%Matrix.

DataBox = CurrData(x-BoxRadius:x+BoxRadius,y-BoxRadius:y+BoxRadius,z-BoxRadius:z+BoxRadius);