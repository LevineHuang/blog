hdfs用户具有写权限目录：/var/lib/hadoop-hdfs/

我在新部署一个MR，oozie workflow中配了一个ssh动作，没执行到这个ssh就报：
Message [UNKOWN_ERROR: Cannot run program "scp": error=2, No such file or directory


手动在命令行执行scp ，能正常把文件copy到远程。
<action name="db-opr">
        <ssh xmlns="uri:oozie:ssh-action:0.1">
            <host>${ssh_user}@${ssh_ip}</host>
            <command>sh ${shellDir}/db/sqlScriptOprByHour.sh ${CURRENT_DATE} ${CURRENT_HOUR}</command>
        </ssh>
        <ok to="sqoop-fork"/>
        <error to="sqoop-fork"/>
    </action>
