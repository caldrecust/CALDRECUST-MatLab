function ExportDesignSSRecBeam(directionData,DimBeamsCollec,db18Spans,...
    LenRebarL,LenRebarM,LenRebarR,DistrRebarLeft,DistrRebarMid,DistrRebarRight,...
    totnbSpan,tenbLMRspan,ListRebarDiamLeft,ListRebarDiamMid,ListRebarDiamRight,...
    diamListdSdb,distrSb,beamNSb,ShearBeamDesignCollec)

%------------------------------------------------------------------------
% Syntax:
% ExportResultsBeam(directionData,dim_beams_collection,coordEndBeams,...
%   disposition_rebar_beams3sec,nbarbeamsCollection,arrangemetbarbeams)
%
%-------------------------------------------------------------------------
% SYSTEM OF UNITS: Any.
%
%------------------------------------------------------------------------
% PURPOSE: Computes the exportation of the design results of a beam element
%          into a .txt file on a prescribed folder route.
% 
% INPUT:  directionData:            is the folder disc location to save the
%                                   results
%
%         dim_beams_collection:     is the array containing the cross-section
%                                   dimensions data of the beam element
%
%         coordEndBeams:            is the array containing the centroid 
%                                   coordinates of the initial cross-
%                                   section's end of the beam
%
%         disposition_rebar_beams3sec: is the array containing the local 
%                                      rebar coordinates of each of the 
%                                      three critical design cross-sections
%                                      of the beam
%
%         nbarbeamsCollection:      is the total number of rebars of each 
%                                   of the three design cross-sections, both 
%                                   in tension and compression. 
%                                   Size = [1,6] in format:
%
%               [nbarsLeft_{ten},nbarsLeft_{comp},nbarsCenter_{ten},...
%               nbarsCenter_{comp},nbarsRight_{ten},nbarsRight_{comp}]
%
%         arrangemetbarbeams:       is the list of all the rebar types used 
%                                   in the element
%
%------------------------------------------------------------------------
% LAST MODIFIED: L.F.Veduzco    2025-02-05
% Copyright (c)  School of Engineering
%                HKUST
%------------------------------------------------------------------------

% Opening files
nombre_archivo='nBarBeamsCollec.csv';
fileid_01=fopen([directionData,nombre_archivo],'w+t');

nombre_archivo='dimBeamsCollec.csv';
fileid_02=fopen([directionData,nombre_archivo],'w+t');

nombre_archivo='DiamBarBeamsCollec3sec.csv';
fileid_03=fopen([directionData,nombre_archivo],'w+t');

nombre_archivo='DistrBarBeams3sec.csv';
fileid_04=fopen([directionData,nombre_archivo],'w+t');

nombre_archivo='LenBarBeams3sec.csv';
fileid_05=fopen([directionData,nombre_archivo],'w+t');

nombre_archivo='NbarsTot3sec.csv';
fileid_06=fopen([directionData,nombre_archivo],'w+t');

nombre_archivo='DiamListSb.csv';
fileid_07=fopen([directionData,nombre_archivo],'w+t');

nombre_archivo='DistrSb.csv';
fileid_08=fopen([directionData,nombre_archivo],'w+t');

nombre_archivo='NSb.csv';
fileid_09=fopen([directionData,nombre_archivo],'w+t');

% To write .txt files for the exportation of results 
nbeams=length(DimBeamsCollec(:,1));

for i=1:nbeams
    fprintf(fileid_01,'%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d\n',tenbLMRspan(i,:));
    fprintf(fileid_02,'%.2f,%.2f,%.2f,%.2f,%.2f\n',DimBeamsCollec(i,:));
end

ListDiamLMR=[ListRebarDiamLeft;ListRebarDiamMid;ListRebarDiamRight];
DistrRebarLMR=[DistrRebarLeft;DistrRebarMid;DistrRebarRight];

for j=1:length(ListDiamLMR(:,1))
    fprintf(fileid_04,'%.2f,%.2f\n',DistrRebarLMR(j,:));
end
fprintf(fileid_03,'%d\n',ListDiamLMR(:,:));
fprintf(fileid_07,'%d\n',diamListdSdb(:,:));

for i=1:length(diamListdSdb)
    fprintf(fileid_08,'%.2f,%.2f\n',distrSb(i,:));
end

LenRebar=[LenRebarL;LenRebarM;LenRebarR];
for j=1:length(LenRebar(:,1))
    fprintf(fileid_05,'%.2f\n',LenRebar(j,:));
end

for i=1:nbeams
    fprintf(fileid_06,'%d,%d,%d\n',totnbSpan(i,:));
    
end
fprintf(fileid_09,'%d\n',beamNSb(:,:));

fclose(fileid_01);
fclose(fileid_02);
fclose(fileid_03);
fclose(fileid_04);
fclose(fileid_05);
fclose(fileid_06);
fclose(fileid_07);
fclose(fileid_08);
fclose(fileid_09);

nombre_archivo='TendbSec.csv';
fileid_10=fopen([directionData,nombre_archivo],'w+t');

for i=1:nbeams
    fprintf(fileid_10,'%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d\n',...
        db18Spans(i,:));
end
fclose(fileid_10);


% Exporting shear design data (if required)

nombre_archivo='ShearBeamDesign.csv';
fileid_11=fopen([directionData,nombre_archivo],'w+t');
for i=1:nbeams
    fprintf(fileid_11,'%.1f,%.1f,%.1f,%.2f,%.2f,%.2f\n',...
        ShearBeamDesignCollec(i,:));
end
fclose(fileid_11);
