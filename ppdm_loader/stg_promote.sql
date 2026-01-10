CREATE TABLE stg.promote_audit (
  batch_id      uniqueidentifier NOT NULL,
  target_schema sysname NOT NULL,
  target_table  sysname NOT NULL,
  action        nvarchar(10) NOT NULL,     -- 'INSERT'/'UPDATE'
  pk_json       nvarchar(max) NULL,
  rid           bigint NULL,
  created_utc   datetime2(3) NOT NULL DEFAULT SYSUTCDATETIME()
);
GO
