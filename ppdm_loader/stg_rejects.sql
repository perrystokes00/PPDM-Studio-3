CREATE TABLE stg.rejects (
  batch_id         uniqueidentifier NOT NULL,
  stage            nvarchar(50) NOT NULL,     -- e.g. 'PROMOTE_PRECHECK'
  target_schema    sysname NOT NULL,
  target_table     sysname NOT NULL,
  reason_code      nvarchar(50) NOT NULL,     -- e.g. 'FK_MISSING', 'PK_NULL', 'SRC_DUP_PK'
  reason_detail    nvarchar(4000) NULL,       -- fk name, parent table, etc
  pk_json          nvarchar(max) NULL,        -- PK values (json)
  fk_json          nvarchar(max) NULL,        -- FK values (json)
  rid              bigint NULL,               -- if you have RID in norm
  created_utc      datetime2(3) NOT NULL DEFAULT SYSUTCDATETIME()
);
GO
