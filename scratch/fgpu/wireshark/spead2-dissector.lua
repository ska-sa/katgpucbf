-- A sort of minimum working example of how to do a Wireshark dissector using the
-- built-in LUA scripting. It's not what I'd call fast, and there may be more
-- elegant ways of doing it, but it works.
-- It's not flexible - i.e. it's hardcoded for the F-engine output the way it's
-- specified in the ICD. If the SPEAD header says there's another number of
-- item fields, then this will be ignored.

spead_proto = Proto("SPEAD","SPEAD Protocol")

-- Best way to understand these fields is to look at the original SPEAD spec
-- (2010 document on CASPER wiki) along with the data subscribers ICD
-- (M1000-0001-020 section 4.4.5.2.2.1.)
local magic_no = ProtoField.uint8("SPEAD.magic_no", "Magic Number", base.HEX)
local version = ProtoField.uint8("SPEAD.version", "Version", base.DEC)
local item_pointer_width = ProtoField.uint8("SPEAD.item_pointer_width", "Item Pointer Width", base.DEC)
local heap_addr_width = ProtoField.uint8("SPEAD.heap_addr_width", "Heap Address Width", base.DEC)
local num_items = ProtoField.uint16("SPEAD.num_items", "Num Items", base.DEC)
local heap_counter = ProtoField.int64("SPEAD.heap_counter", "Heap Counter", base.DEC)
local heap_size = ProtoField.int64("SPEAD.heap_size", "Heap Size", base.DEC)
local heap_offset = ProtoField.int64("SPEAD.heap_offset", "Heap Offset", base.DEC)
local packet_payload_length = ProtoField.int64("SPEAD.packet_payload_length", "Packet Payload Length", base.DEC)
local timestamp = ProtoField.int64("SPEAD.timestamp", "Timestamp", base.DEC)
local feng_id = ProtoField.int64("SPEAD.feng_id", "FEng ID", base.DEC)
local frequency = ProtoField.int64("SPEAD.frequency", "Frequency", base.DEC)
local feng_data_id = ProtoField.int64("SPEAD.feng_data_id", "FEng Data addr", base.DEC)
local field_9 = ProtoField.int64("SPEAD.field_9", "Field 9", base.DEC)
local field_10 = ProtoField.int64("SPEAD.field_10", "Field 10", base.DEC)
local field_11 = ProtoField.int64("SPEAD.field_11", "Field 11", base.DEC)
local feng_raw = ProtoField.bytes("SPEAD2.feng_raw", "FEng raw", base.COLON)

spead_proto.fields = {
	magic_no,
	version,
	item_pointer_width,
	heap_addr_width,
	num_items,
	heap_counter,
	heap_size,
	heap_offset,
	packet_payload_length,
	timestamp,
	feng_id,
	frequency,
	feng_data_id,
	field_9,
	field_10,
	field_11,
	feng_raw,
}

function spead_proto.dissector(buffer,pinfo,tree)
	pinfo.cols.protocol = "SPEAD"
	local subtree = tree:add(spead_proto,buffer(),"SPEAD Protocol Data")

	subtree:add(magic_no, buffer(0,1))
	subtree:add(version, buffer(1,1))
	subtree:add(item_pointer_width, buffer(2,1))
	subtree:add(heap_addr_width, buffer(3,1))
	subtree:add(num_items, buffer(6,2))
    -- These are all only 6 bytes because the first two are just the item number:
    -- like 0x1600 for timestamp and 0x4101 for fengine ID. It's easier just to
    -- ignore those first two bytes, then you get the actual number.
	subtree:add(heap_counter, buffer(10,6))
	subtree:add(heap_size, buffer(18,6))
	subtree:add(heap_offset, buffer(26,6))
	subtree:add(packet_payload_length, buffer(34,6))
	subtree:add(timestamp, buffer(42,6))
	subtree:add(feng_id, buffer(50,6))
	subtree:add(frequency, buffer(58,6))
	subtree:add(feng_data_id, buffer(66,6))
	subtree:add(field_9, buffer(74,6))
	subtree:add(field_10, buffer(82,6))
	subtree:add(field_11, buffer(90,6))
	subtree:add(feng_raw, buffer(96,buffer:len()-96))
end

udp_table = DissectorTable.get("udp.port")
-- We've commonly used these two ports, 7148 is the usual one, 7149 is used
-- by src/tools/fsim. Adjust according to your needs.
udp_table:add(7148,spead_proto)
udp_table:add(7149,spead_proto)
