/* ---------- cfg schema ---------- */
IF SCHEMA_ID('cfg') IS NULL
    EXEC('CREATE SCHEMA cfg');
GO

/* ---------- stg schema ---------- */
IF SCHEMA_ID('stg') IS NULL
    EXEC('CREATE SCHEMA stg');
GO

/* ---------- cfg.etl_rule_def ---------- */
IF OBJECT_ID('cfg.etl_rule_def','U') IS NULL
BEGIN
    CREATE TABLE cfg.etl_rule_def(
        rule_id        varchar(100)   NOT NULL CONSTRAINT PK_etl_rule_def PRIMARY KEY,
        [description]  nvarchar(800)  NULL,
        domain         varchar(50)    NULL,
        phase          varchar(20)    NULL,
        enabled        bit            NOT NULL CONSTRAINT DF_etl_rule_def_enabled DEFAULT(1),
        rule_type      varchar(30)    NOT NULL,
        column_name    sysname        NOT NULL,
        severity       varchar(10)    NOT NULL,
        error_message  nvarchar(800)  NULL,
        params_json    nvarchar(max)  NULL,
        updated_at     datetime2(0)   NOT NULL CONSTRAINT DF_etl_rule_def_updated DEFAULT(sysdatetime())
    );
END
GO

/* ---------- cfg.rule_run ---------- */
IF OBJECT_ID('cfg.rule_run','U') IS NULL
BEGIN
    CREATE TABLE cfg.rule_run(
        run_id      bigint IDENTITY(1,1) NOT NULL CONSTRAINT PK_rule_run PRIMARY KEY,
        started_at  datetime2(0) NOT NULL CONSTRAINT DF_rule_run_started DEFAULT(sysdatetime()),
        finished_at datetime2(0) NULL,
        domain      varchar(50) NULL,
        phase       varchar(20) NULL,
        note        nvarchar(400) NULL
    );
END
GO

/* ---------- cfg.rule_issue ---------- */
IF OBJECT_ID('cfg.rule_issue','U') IS NULL
BEGIN
    CREATE TABLE cfg.rule_issue(
        issue_id    bigint IDENTITY(1,1) NOT NULL CONSTRAINT PK_rule_issue PRIMARY KEY,
        run_id      bigint NULL,
        RID         int NOT NULL,
        rule_id     varchar(100) NOT NULL,
        column_name sysname NULL,
        severity    varchar(10) NOT NULL,
        message     nvarchar(800) NULL,
        created_at  datetime2(0) NOT NULL CONSTRAINT DF_rule_issue_created DEFAULT(sysdatetime()),
        CONSTRAINT FK_rule_issue_run FOREIGN KEY (run_id) REFERENCES cfg.rule_run(run_id)
    );
END
GO

/* ---------- stg.invalid_rows ---------- */
IF OBJECT_ID('stg.invalid_rows','U') IS NULL
BEGIN
    CREATE TABLE stg.invalid_rows(
        RID         int NOT NULL,
        rule_id     varchar(100) NOT NULL,
        column_name sysname NULL,
        severity    varchar(10) NOT NULL,
        message     nvarchar(800) NULL
    );
END
ELSE
    TRUNCATE TABLE stg.invalid_rows;
GO

/* ---------- stg.valid_rid ---------- */
IF OBJECT_ID('stg.valid_rid','U') IS NULL
BEGIN
    CREATE TABLE stg.valid_rid(
        RID int NOT NULL CONSTRAINT PK_valid_rid PRIMARY KEY
    );
END
ELSE
    TRUNCATE TABLE stg.valid_rid;
GO
