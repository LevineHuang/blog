- /gddr/output/stat-data/day/20160529/icphostiptopn/usergroup_*/part-r-*

copy
( select host,hostip::inet as ip,round(sum(upbyte)*8/1024/1024/86400,4) as up ,round(sum(downbyte)*8/1024/1024/86400,4)  as  down,round(sum(upbyte+downbyte)*8/1024/1024/86400,2) as sum
from     td_usergrp_host_d_20160525
 where domainid=119
and  servid in (566,569,573,575,579,442)
and host in (select host from td_host_temp)
group by host,ip order by down desc )
 to '/home/tmp/gdurlhost20160525.csv' csv ;

copy (select host,round(sum(upbyte)*8/1024/1024/86400,4) as up,
round(sum(downbyte)*8/1024/1024/86400,4) as down,
round(sum(upbyte+downbyte)*8/1024/1024/86400,2) as sum,
sum(count) as cnt from td_usergrp_host_d_20160509
where domainid=119 and host in (select inhost from tmp_host)
group by host order by down desc) to '/home/tmp/hosts1.csv' csv;
